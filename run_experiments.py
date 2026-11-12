"""Run the experiment set behind RESULTS.md and the filled implementation document.

Each experiment writes to its own folder under results/. Finished experiments (results.json
present) are skipped, so the script can be stopped and restarted.

  python scripts/run_experiments.py                # everything
  python scripts/run_experiments.py --only areas   # one group: main | eval | areas | ablations | baselines
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = ["--data", "primekg", "--smiles-csv", "data/primekg/drug_smiles.csv", "--epochs", "60"]
AREAS = ["neurodegenerative", "mental_health", "cardiovascular", "metabolic_disorder", "autoimmune",
         "cell_proliferation", "anemia", "adrenal_gland", "diabetes"]
BASELINES = ["rgcn", "compgcn", "hgt", "transe", "distmult", "rotate"]
ABLATIONS = {
    "A1_no_tricals": ["--no-tricals"],
    "A2_no_modality_dropout": ["--no-modality-dropout"],
    "A3_no_subtype_conditioning": ["--no-film"],
    "A6_random_negatives_only": ["--random-negatives-only"],
}


def experiments():
    exps = {
        # Main run: S1 + S2 (neurodegenerative), AD and long COVID case studies, REX.
        "main": ("primekg", ["--splits", "s1", "s2", "--area", "neurodegenerative", "--case-disease",
                             "Alzheimer disease", "long COVID-19", "--explain-n", "5"]),
        # Re-evaluate the main checkpoints to export raw scores (no retraining, no REX).
        "eval": ("primekg_eval", ["--splits", "s1", "s2", "--area", "neurodegenerative", "--resume", "--explain-n", "0"]),
    }
    for a in AREAS[1:]:
        exps[f"areas:{a}"] = (f"areas/{a}", ["--splits", "s2", "--area", a, "--explain-n", "0"])
    for name, flags in ABLATIONS.items():
        exps[f"ablations:{name}"] = (f"ablations/{name}", ["--splits", "s1", "s2", "--area", "neurodegenerative",
                                                           "--explain-n", "0", *flags])
    for enc in BASELINES:
        exps[f"baselines:{enc}"] = (f"baselines/{enc}", ["--splits", "s1", "s2", "--area", "neurodegenerative",
                                                          "--explain-n", "0", "--encoder", enc])
    exps["baselines:llm_ranking"] = ("baselines/llm_ranking", None)
    return exps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["main", "eval", "areas", "ablations", "baselines"])
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to run.py")
    a = p.parse_args()
    for key, (out, args) in experiments().items():
        if a.only and key.split(":")[0] != a.only:
            continue
        out_dir = os.path.join(ROOT, "results", out)
        if os.path.exists(os.path.join(out_dir, "summary.md")):
            print(f"[skip] {key} (done)")
            continue
        if key == "eval":  # reuse the main checkpoints
            os.makedirs(out_dir, exist_ok=True)
            for f in ("model_s1_seed0.pt", "model_s2_seed0.pt", "results.json"):
                src = os.path.join(ROOT, "results", "primekg", f)
                if os.path.exists(src) and not os.path.exists(os.path.join(out_dir, f)):
                    shutil.copy(src, out_dir)
        if args is None:  # LLM-only ranking has its own script
            cmd = [sys.executable, "scripts/llm_baseline.py", "--out", f"results/{out}"]
        else:
            cmd = [sys.executable, "scripts/run.py", *COMMON, *args, "--out", f"results/{out}", *a.extra]
        print(f"[run] {key}: {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
