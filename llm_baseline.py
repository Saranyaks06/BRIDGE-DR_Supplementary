"""LLM-only ranking baseline: a local LLM scores drug-disease pairs without the knowledge graph.

Uses the same split and the same negative construction as BRIDGE-DR's AUROC/AUPRC (one random
drug per positive plus held-out contraindications), on a sample of test positives. Ranking
metrics (Hits@k, MRR) would need every drug scored for every disease and are not computed.

  python scripts/llm_baseline.py --split s1 --n-pos 150 --out results/baselines/llm_ranking
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge_dr.data import load_primekg, make_split  # noqa: E402
from bridge_dr.metrics import classification  # noqa: E402

PROMPT = ("You are a pharmacology expert. On a scale from 0 to 100, how likely is it that the drug "
          "\"{drug}\" is an approved or effective treatment for the disease \"{disease}\"? "
          "Answer with a single integer only.")


def ask(model: str, drug: str, disease: str) -> float:
    body = json.dumps({"model": model, "prompt": PROMPT.format(drug=drug, disease=disease), "stream": False,
                       "options": {"temperature": 0, "num_predict": 6}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    text = json.loads(urllib.request.urlopen(req, timeout=120).read())["response"]
    m = re.search(r"\d+(\.\d+)?", text)
    return min(float(m.group()), 100.0) if m else 50.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-path", default="data/primekg/kg.csv")
    p.add_argument("--splits", nargs="+", default=["s1", "s2"])
    p.add_argument("--area", default="neurodegenerative")
    p.add_argument("--n-pos", type=int, default=150)
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/baselines/llm_ranking")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load_primekg(a.kg_path)
    names = data.node_names
    results = []
    for kind in a.splits:
        split = make_split(data, kind, a.seed, area=a.area)
        g = torch.Generator().manual_seed(a.seed)
        pos = split.test_pos[torch.randperm(len(split.test_pos), generator=g)[: a.n_pos]]
        known = set(map(tuple, data.indication.tolist()))
        rnd = []
        while len(rnd) < len(pos):
            d = int(data.drug_nodes[torch.randint(len(data.drug_nodes), (1,), generator=g)])
            c = int(pos[len(rnd), 1])
            if (d, c) not in known:
                rnd.append((d, c))
        hard = split.test_null[torch.randperm(len(split.test_null), generator=g)[: len(pos)]].tolist()
        pairs = [tuple(x) for x in pos.tolist()] + rnd + [tuple(x) for x in hard]
        y = np.r_[np.ones(len(pos)), np.zeros(len(rnd) + len(hard))]
        t0, scores = time.time(), []
        for i, (d, c) in enumerate(pairs):
            scores.append(ask(a.model, names[d], names[c]))
            if i % 50 == 0:
                print(f"[{kind}] {i}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        test = classification(y, np.array(scores))
        print(f"[{kind}] {json.dumps(test)}")
        results.append({"split": kind, "seed": a.seed, "test": test,
                        "fit": {"train_seconds": 0.0, "peak_vram_gb": 0.0, "inference_seconds": time.time() - t0},
                        "llm": a.model, "n_pairs": len(pairs), "n_pos": len(pos)})
        with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{r['split']}: AUROC {r['test']['auroc']:.4f} AUPRC {r['test']['auprc']:.4f} "
                          f"({r['n_pairs']} pairs, {r['llm']})" for r in results))


if __name__ == "__main__":
    main()
