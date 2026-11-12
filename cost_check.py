"""Peak RAM and CPU-only timings for Table 14.

Each component runs in its own process so that its peak RAM is measured on its own:
  train      one training epoch of BRIDGE-DR (S1) on the chosen device
  inference  embed the graph and score all 7,957 drugs for one disease
  cfs        conformal p-values + Benjamini-Hochberg over all candidates of the S1 test set
  rex        one REX explanation (path search, PubMed, Ollama LLM, NLI on CPU); RAM of this process only

  python scripts/cost_check.py --device cpu     # CPU-only column
  python scripts/cost_check.py --device cuda    # peak RAM next to the GPU timings
Writes reports/figure_data/table14_cost_measured_<device>.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def component(name, device):
    import numpy as np
    import torch

    from bridge_dr.cfs import cfs_select
    from bridge_dr.data import load_primekg, make_split
    from bridge_dr.trainer import Trainer, TrainConfig, peak_ram_gb

    data = load_primekg(os.path.join(ROOT, "data/primekg/kg.csv"), 150_000,
                        smiles_csv=os.path.join(ROOT, "data/primekg/drug_smiles.csv"))
    split = make_split(data, "s1", 0)
    tr = Trainer(data, split, TrainConfig(), device, 0)
    tr.model.load_state_dict(torch.load(os.path.join(ROOT, "results/primekg/model_s1_seed0.pt"), map_location=device))
    t0 = time.time()
    if name == "train":
        tp = split.train_pos[torch.randperm(len(split.train_pos))]
        for i in range(0, len(tp), tr.cfg.batch_size):
            tr.train_step(tp[i:i + tr.cfg.batch_size])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        out = {"seconds_per_epoch": time.time() - t0}
    elif name == "inference":
        with torch.no_grad():
            h, _, _ = tr.embed()
            tr.drug_scores(h, int(split.test_pos[0, 1]))
        out = {"seconds": time.time() - t0}
    elif name == "rex":
        from bridge_dr.rex import Explainer
        full_ei, full_et = tr.graph(None)
        rex = Explainer(data, full_ei.cpu(), full_et.cpu(), torch.ones(full_ei.shape[1]), None, rel_names=tr.rel_names)
        with open(os.path.join(ROOT, "results/primekg/explanations_s1.json"), encoding="utf-8") as fh:
            e = json.load(fh)[0]
        t0 = time.time()
        rex.explain(e["paths"][0]["node_ids"][0], e["paths"][0]["node_ids"][-1])
        out = {"seconds": time.time() - t0}
    else:  # cfs
        with torch.no_grad():
            h, _, _ = tr.embed()
            cal = tr.score(h, split.calib_null).cpu().numpy()
            cand = torch.cat([split.test_pos, split.test_null])
            cs = tr.score(h, cand).cpu().numpy()
        t0 = time.time()
        area = lambda p: np.array([data.disease_area.get(int(c), "other") for c in p[:, 1]])  # noqa: E731
        cfs_select(cs, area(cand), cal, area(split.calib_null), 0.10)
        out = {"seconds": time.time() - t0}
    out["peak_ram_gb"] = peak_ram_gb()
    out["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--component")
    ap.add_argument("--components", nargs="+", default=["train", "inference", "cfs"])
    a = ap.parse_args()
    if a.component:
        return component(a.component, a.device)
    import pandas as pd
    rows = []
    for name in a.components:
        r = subprocess.run([sys.executable, __file__, "--device", a.device, "--component", name],
                           capture_output=True, text=True, cwd=ROOT)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if not line:
            print(r.stderr[-2000:])
            continue
        rows.append({"component": name, "device": a.device, **json.loads(line[-1])})
        print(rows[-1], flush=True)
    out = os.path.join(ROOT, "reports", "figure_data", f"table14_cost_measured_{a.device}.csv")
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        new = pd.concat([old[~old["component"].isin(new["component"])], new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
