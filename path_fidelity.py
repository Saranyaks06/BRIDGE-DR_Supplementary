"""Path fidelity for Table 12: how much a prediction depends on the KG paths given as its explanation.

For each prediction explained in the main run (results/primekg/explanations_{s1,s2}.json), the edges
of the explanation's top-5 paths are removed from the message-passing graph and the drug-disease
score is recomputed. Fidelity = (p_before - p_after) / p_before, with p the sigmoid of the score,
averaged over predictions that have at least one path.

  rex:       paths ranked by attention x gradient saliency (REX)
  path_only: the same path search without model weights (paths in graph order)
  random:    the same number of random edges at the drug node (control)

  python scripts/path_fidelity.py --device cpu --out results/path_fidelity
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge_dr.data import load_primekg, make_split  # noqa: E402
from bridge_dr.rex import Explainer  # noqa: E402
from bridge_dr.trainer import Trainer, TrainConfig  # noqa: E402


def path_edges(paths):
    e = set()
    for p in paths:
        ids = p["node_ids"]
        for a, b in zip(ids[:-1], ids[1:]):
            e |= {(a, b), (b, a)}
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/primekg")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/path_fidelity")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load_primekg("data/primekg/kg.csv", 150_000, smiles_csv="data/primekg/drug_smiles.csv")
    rng = np.random.default_rng(0)
    rows = []
    for kind in ("s1", "s2"):
        exp_p = os.path.join(a.results, f"explanations_{kind}.json")
        if not os.path.exists(exp_p):
            continue
        split = make_split(data, kind, 0, area="neurodegenerative")
        tr = Trainer(data, split, TrainConfig(), a.device, 0)
        tr.model.load_state_dict(torch.load(os.path.join(a.results, f"model_{kind}_seed0.pt"), map_location=a.device))
        tr.model.eval()
        full_ei, full_et = tr.graph(None)
        plain = Explainer(data, full_ei.cpu(), full_et.cpu(), torch.ones(full_ei.shape[1]), None, nli_model=None,
                          use_pubmed=False, rel_names=tr.rel_names)
        with torch.no_grad():
            h, ei, et = tr.embed()
            src, dst = ei[0].cpu().numpy(), ei[1].cpu().numpy()
            keys = src.astype(np.int64) * data.num_nodes + dst
            for e in json.load(open(exp_p, encoding="utf-8")):
                if not e["paths"]:
                    continue
                drug, dis = e["paths"][0]["node_ids"][0], e["paths"][0]["node_ids"][-1]
                pair = torch.tensor([[drug, dis]])
                p0 = torch.sigmoid(tr.score(h, pair)).item()
                sets = {"rex": path_edges(e["paths"]), "path_only": path_edges(plain.paths(drug, dis))}
                n_cut = int(np.isin(keys, [u * data.num_nodes + v for u, v in sets["rex"]]).sum())
                at_drug = np.where((src == drug) | (dst == drug))[0]
                rnd = rng.choice(at_drug, size=min(max(n_cut, 1), len(at_drug)), replace=False)
                row = {"split": kind, "drug": e["drug"], "disease": e["disease"], "p_before": p0}
                for name in ("rex", "path_only", "random"):
                    if name == "random":
                        mask = np.ones(len(keys), bool)
                        mask[rnd] = False
                    else:
                        mask = ~np.isin(keys, [u * data.num_nodes + v for u, v in sets[name]])
                    m = torch.from_numpy(mask).to(ei.device)
                    h2, _ = tr.encode(ei[:, m], et[m])
                    p1 = torch.sigmoid(tr.score(h2, pair)).item()
                    row[f"{name}_edges_removed"] = int((~mask).sum())
                    row[f"{name}_fidelity"] = (p0 - p1) / p0 if p0 > 0 else 0.0
                rows.append(row)
                print(f"[{kind}] {e['drug']} -> {e['disease']}: p {p0:.3f}, fidelity rex {row['rex_fidelity']:.3f} "
                      f"path-only {row['path_only_fidelity']:.3f} random {row['random_fidelity']:.3f}", flush=True)
        del tr, h
    summary = {k: float(np.mean([r[f"{k}_fidelity"] for r in rows])) for k in ("rex", "path_only", "random")} if rows else {}
    summary["n_predictions"] = len(rows)
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
