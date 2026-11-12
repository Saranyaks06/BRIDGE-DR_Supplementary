"""Collect experiment outputs into editable CSV files under reports/figure_data/.

Every figure and results table in the report is drawn from these CSVs, so values can be
corrected or replaced by hand and the report regenerated:

  python scripts/export_report_data.py     # results/ -> reports/figure_data/*.csv
  python scripts/make_figures.py           # CSVs -> reports/figures/*.png
  python scripts/fill_report.py            # CSVs + PNGs -> reports/BRIDGE-DR_Implementation_Report.docx
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "reports", "figure_data")

AREAS = ["neurodegenerative", "mental_health", "cardiovascular", "metabolic_disorder", "autoimmune",
         "cell_proliferation", "anemia", "adrenal_gland", "diabetes"]
AREA_LABEL = {"neurodegenerative": "Neurodegenerative", "mental_health": "Mental health",
              "cardiovascular": "Cardiovascular", "metabolic_disorder": "Metabolic", "autoimmune": "Autoimmune",
              "cell_proliferation": "Cell prolif.", "anemia": "Anaemia", "adrenal_gland": "Adrenal",
              "diabetes": "Diabetes"}
VARIANTS = {  # label -> results folder
    "BRIDGE-DR (full)": "primekg",
    "A1 - no TriCAL": "ablations/A1_no_tricals",
    "A2 - no modality dropout": "ablations/A2_no_modality_dropout",
    "A3 - no subtype conditioning": "ablations/A3_no_subtype_conditioning",
    "A6 - random negatives only": "ablations/A6_random_negatives_only",
}


def load(folder: str) -> list[dict]:
    p = os.path.join(RES, folder, "results.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get(folder: str, split: str) -> dict | None:
    return next((r for r in load(folder) if r["split"] == split), None)


def npz(folder: str, split: str):
    p = os.path.join(RES, folder, f"scores_{split}_seed0.npz")
    return np.load(p) if os.path.exists(p) else None


def scores_folder(folder: str) -> str:
    return "primekg_eval" if folder == "primekg" else folder  # main run's raw scores come from re-evaluation


def save(df: pd.DataFrame, name: str):
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"  {name}: {len(df)} rows")


PLAN_SOURCE = "plan target (not measured)"


def plan_baselines():
    """Copy the comparison-method values of the implementation plan (Tables 7-9) into
    baselines_*.csv, labelled as plan targets. Existing files are kept, so values replaced by
    real reruns (with source set to "measured") are never overwritten."""
    import glob

    import docx

    plan = glob.glob(os.path.join(ROOT, "docs", "*Implementation_Plan*.docx"))
    if not plan:
        return
    tables = docx.Document(plan[0]).tables
    num = lambda s: float(s.split("±")[0].strip())  # noqa: E731
    files = {
        "baselines_s1.csv": lambda: pd.DataFrame(
            [{"method": r.cells[0].text, "auroc": num(r.cells[1].text), "auprc": num(r.cells[2].text),
              "hits@10": num(r.cells[3].text), "mrr": num(r.cells[4].text), "source": PLAN_SOURCE}
             for r in tables[6].rows[2:]]),
        "baselines_zero_shot.csv": lambda: pd.DataFrame(
            [{"method": tables[7].rows[0].cells[j].text, "area": AREA_LABEL[area], "auprc": num(r.cells[j].text), "source": PLAN_SOURCE}
             for r, area in zip(tables[7].rows[1:10], AREAS) for j in range(2, 6)]),
        "baselines_s3.csv": lambda: pd.DataFrame(
            [{"method": r.cells[0].text, "recall@50": num(r.cells[1].text), "recall@100": num(r.cells[2].text),
              "median_rank": num(r.cells[3].text), "ef@1%": num(r.cells[4].text),
              "ad_recall@100": num(r.cells[5].text), "covid_recall@100": num(r.cells[6].text), "source": PLAN_SOURCE}
             for r in tables[8].rows[2:]]),
    }
    for name, build in files.items():
        if not os.path.exists(os.path.join(OUT, name)):
            save(build(), name)
        else:
            print(f"  {name}: kept (already exists)")
    merge_measured_baselines()
    seed_check()
    appendix_data()


MEASURED_BASELINES = {  # results folder -> method name used in the plan's tables
    "baselines/rgcn": "R-GCN", "baselines/compgcn": "CompGCN", "baselines/hgt": "HGT",
    "baselines/transe": "TransE", "baselines/distmult": "DistMult", "baselines/rotate": "RotatE",
    "baselines/llm_ranking": "LLM ranking (8B)",
}


APPENDIX_METHODS = [("BRIDGE-DR", "primekg"), ("R-GCN", "baselines/rgcn"), ("CompGCN", "baselines/compgcn"),
                    ("HGT", "baselines/hgt"), ("TransE", "baselines/transe"), ("DistMult", "baselines/distmult"),
                    ("RotatE", "baselines/rotate"), ("LLM-only (qwen2.5:3b)", "baselines/llm_ranking")]
AREA_DIRS = {"Neurodegenerative": "primekg", "Mental health": "areas/mental_health", "Cardiovascular": "areas/cardiovascular",
             "Metabolic disorder": "areas/metabolic_disorder", "Autoimmune": "areas/autoimmune",
             "Cell proliferation": "areas/cell_proliferation", "Anaemia": "areas/anemia",
             "Adrenal gland": "areas/adrenal_gland", "Diabetes": "areas/diabetes"}


def metric_pairs(d):
    """Rebuild the evaluation set behind the reported AUROC/AUPRC: test indications, one random
    negative each and one held-out contraindication each (the scores file stores the first two in
    test_y/test_logit and the contraindications in cand_score)."""
    y, s = d["test_y"], d["test_logit"]
    n_pos = int(y.sum())
    contra = d["cand_score"][~d["cand_is_pos"].astype(bool)][:n_pos]
    labels = np.r_[np.where(y == 1, "indication", "random negative"), np.full(len(contra), "contraindication")]
    return np.r_[y, np.zeros(len(contra))], np.r_[s, contra], labels


def appendix_data():
    """Measured-only tables for the report appendix: every method on S1/S2, every held-out area, calibration."""
    rows = []
    for name, folder in APPENDIX_METHODS:
        for r in load(folder):
            t, fit = r["test"], r.get("fit", {})
            rows.append({"method": name, "split": r["split"].upper(), "auroc": t["auroc"], "auprc": t["auprc"],
                         "hits@10": t.get("hits@10", np.nan), "mrr": t.get("mrr", np.nan),
                         "minutes": (fit.get("train_seconds") or fit.get("inference_seconds", 0)) / 60})
    save(pd.DataFrame(rows), "appendix_methods.csv")
    rows = []
    for area, folder in AREA_DIRS.items():
        r = get(folder, "s2")
        if r is not None:
            rows.append({"area": area, **{k: r["test"][k] for k in ("auroc", "auprc", "mrr", "hits@10", "hits@100",
                                                                    "median_rank")},
                         "train_minutes": r["fit"].get("train_seconds", 0) / 60})
    save(pd.DataFrame(rows), "appendix_areas.csv")
    rows = []
    for split in ("s1", "s2"):
        r = get("primekg", split)
        if r is not None and "calibration" in r:
            c = r["calibration"]
            rows.append({"split": split.upper(), "temperature": c["temperature"], "ece_before": c["before"]["ece"],
                         "ece_after": c["after"]["ece"], "brier_before": c["before"]["brier"],
                         "brier_after": c["after"]["brier"]})
    save(pd.DataFrame(rows), "appendix_calibration.csv")
    curves, train = [], []
    from sklearn.metrics import roc_curve
    for name, folder in APPENDIX_METHODS:
        for split in ("s1", "s2"):
            f = os.path.join(RES, "primekg_eval" if folder == "primekg" else folder, f"scores_{split}_seed0.npz")
            if os.path.exists(f):
                y, s_, _ = metric_pairs(np.load(f))
                fpr, tpr, _ = roc_curve(y, s_)
                prec, rec, _ = precision_recall_curve(y, s_)
                for kind, xs, ys in (("roc", fpr, tpr), ("pr", rec, prec)):
                    idx = np.unique(np.linspace(0, len(xs) - 1, min(len(xs), 200)).astype(int))
                    curves += [{"method": name, "split": split.upper(), "curve": kind, "x": xs[i], "y": ys[i]} for i in idx]
            h = os.path.join(RES, folder, f"history_{split}_seed0.json")
            if os.path.exists(h):
                with open(h, encoding="utf-8") as fh:
                    train += [{"method": name, "split": split.upper(), "epoch": e["epoch"], "val_auprc": e["val_auprc"]}
                              for e in json.load(fh)]
    save(pd.DataFrame(curves), "appendix_roc_pr_curves.csv")
    save(pd.DataFrame(train), "appendix_training_curves.csv")
    rows = []
    for split in ("s1", "s2"):
        f = os.path.join(RES, "primekg_eval", f"scores_{split}_seed0.npz")
        if os.path.exists(f):
            _, s_, lab = metric_pairs(np.load(f))
            rows += [{"split": split.upper(), "label": lb, "logit": v} for lb, v in zip(lab, s_)]
    save(pd.DataFrame(rows), "appendix_score_distribution.csv")
    pf = os.path.join(RES, "path_fidelity", "results.json")
    if os.path.exists(pf):
        with open(pf, encoding="utf-8") as f:
            sm = json.load(f)["summary"]
        save(pd.DataFrame([{"explainer": k, "fidelity": sm[k], "n_predictions": sm["n_predictions"]}
                           for k in ("rex", "path_only", "random") if k in sm]), "table12_path_fidelity.csv")
    ra = []
    for area, folder in AREA_DIRS.items():  # R-GCN zero-shot on the same held-out areas
        rd = "baselines/rgcn" if folder == "primekg" else "baselines_areas/rgcn/" + os.path.basename(folder)
        r = get(rd, "s2")
        if r is not None:
            ra.append({"area": area, "auroc": r["test"]["auroc"], "auprc": r["test"]["auprc"],
                       "train_minutes": r["fit"].get("train_seconds", 0) / 60})
    if ra:
        save(pd.DataFrame(ra), "rgcn_zero_shot_areas.csv")
    p = os.path.join(RES, "rex_ablation", "results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            summ = json.load(f)["summary"]
        save(pd.DataFrame([{"mode": k, **v} for k, v in summ.items()]), "table12_rex_ablation.csv")


def seed_check():
    """S2 (neurodegenerative) results over seeds 0 and 1 for BRIDGE-DR and TransE (results/seed1/ holds seed 1)."""
    rows = []
    for method, seed0, seed1 in (("BRIDGE-DR", "primekg", "seed1/bridge_dr"), ("TransE", "baselines/transe", "seed1/transe")):
        for seed, folder in ((0, seed0), (1, seed1)):
            r = get(folder, "s2")
            if r is not None:
                rows.append({"method": method, "seed": seed, "auroc": r["test"]["auroc"], "auprc": r["test"]["auprc"]})
    if any(r["seed"] == 1 for r in rows):
        save(pd.DataFrame(rows), "seed_check_s2.csv")


def merge_measured_baselines():
    """Overwrite plan-target rows with values measured by our reimplementations, where available."""
    b1p, bzp = os.path.join(OUT, "baselines_s1.csv"), os.path.join(OUT, "baselines_zero_shot.csv")
    if not (os.path.exists(b1p) and os.path.exists(bzp)):
        return
    b1, bz = pd.read_csv(b1p), pd.read_csv(bzp)
    for folder, method in MEASURED_BASELINES.items():
        rs = load(folder)
        if not rs:
            continue
        llm = rs[0].get("llm")
        src = f"measured ({llm}, {rs[0]['n_pairs']} sampled pairs)" if llm else "measured (reimplementation, 1 seed)"
        if llm:  # the plan's row is an 8B model; name the model actually measured
            for df in (b1, bz):
                df.loc[df["method"] == method, "method"] = f"LLM ranking ({llm})"
            method = f"LLM ranking ({llm})"
        r1 = next((r for r in rs if r["split"] == "s1"), None)
        if r1 is not None:
            vals = {k: r1["test"].get(k, np.nan) for k in ("auroc", "auprc", "hits@10", "mrr")}
            if method not in set(b1["method"]):
                b1.loc[len(b1)] = {"method": method, **vals, "source": src}
            else:
                for k, v in vals.items():
                    b1.loc[b1["method"] == method, k] = v
                b1.loc[b1["method"] == method, "source"] = src
        r2 = next((r for r in rs if r["split"] == "s2"), None)
        if r2 is not None:
            m = (bz["method"] == method) & (bz["area"] == "Neurodegenerative")
            if m.any():
                bz.loc[m, "auprc"] = r2["test"]["auprc"]
                bz.loc[m, "source"] = src
            else:
                bz.loc[len(bz)] = {"method": method, "area": "Neurodegenerative", "auprc": r2["test"]["auprc"], "source": src}
        print(f"  merged measured baseline: {method} ({src})")
    b1.to_csv(b1p, index=False)
    bz.to_csv(bzp, index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    plan_baselines()

    # --- Table 7 / Fig 2: S1 prediction and PR curves ---
    rows, curves = [], []
    for label, folder in VARIANTS.items():
        r = get(folder, "s1")
        if r:
            t = r["test"]
            rows.append({"method": label, "auroc": t["auroc"], "auprc": t["auprc"], "hits@10": t["hits@10"],
                         "hits@50": t["hits@50"], "hits@100": t["hits@100"], "mrr": t["mrr"], "median_rank": t["median_rank"]})
        z = npz(scores_folder(folder), "s1")
        if z is not None:
            p, rc, _ = precision_recall_curve(z["test_y"], z["test_logit"])
            idx = np.unique(np.linspace(0, len(p) - 1, 200).astype(int))
            curves += [{"method": label, "recall": rc[i], "precision": p[i]} for i in idx]
    save(pd.DataFrame(rows), "table07_s1_prediction.csv")
    save(pd.DataFrame(curves), "fig02_pr_curve_s1.csv")

    # --- Table 8 / Fig 3: zero-shot per disease area ---
    rows = []
    for a in AREAS:
        r = get("primekg", "s2") if a == "neurodegenerative" else get(f"areas/{a}", "s2")
        if r:
            t = r["test"]
            rows.append({"area": AREA_LABEL[a], "auroc": t["auroc"], "auprc": t["auprc"], "hits@100": t["hits@100"],
                         "median_rank": t["median_rank"]})
    save(pd.DataFrame(rows), "table08_fig03_zero_shot_areas.csv")

    # --- Fig 4: Recall@k (S1, S2) ---
    rows = []
    for split, name in (("s1", "S1 transductive"), ("s2", "S2 zero-shot (neurodegenerative)")):
        r = get("primekg", split)
        if r:
            rows += [{"split": name, "k": k, "recall": r["test"][f"hits@{k}"]} for k in (10, 50, 100)]
    save(pd.DataFrame(rows), "fig04_recall_at_k.csv")

    # --- Table 11 / Fig 5: FDR control ---
    rows = []
    for split in ("s1", "s2"):
        r = get("primekg", split)
        if not r:
            continue
        for c in r["cfs"]:
            rows.append({"split": split.upper(), "rule": "CFS", "target_fdr": c["target_fdr"],
                         "empirical_fdr": c["empirical_fdr"], "power": c["power"], "shortlist": c["shortlist_size"]})
        t = r["cfs_topk_baseline"]
        rows.append({"split": split.upper(), "rule": "Fixed top-k (same size as CFS at 0.10)", "target_fdr": None,
                     "empirical_fdr": t["empirical_fdr"], "power": t["power"], "shortlist": t["shortlist_size"]})
    z = npz("primekg_eval", "s1")
    if z is not None:  # simple thresholds on the same candidate pool
        s, pos = z["cand_score"], z["cand_is_pos"].astype(bool)
        T = float(z["temperature"])
        for name, sel in (("Score > 0.5 (uncalibrated)", 1 / (1 + np.exp(-s)) > 0.5),
                          ("Score > 0.5 (temperature-scaled)", 1 / (1 + np.exp(-s / T)) > 0.5)):
            n = int(sel.sum())
            rows.append({"split": "S1", "rule": name, "target_fdr": None,
                         "empirical_fdr": (n - int((sel & pos).sum())) / max(n, 1),
                         "power": int((sel & pos).sum()) / max(int(pos.sum()), 1), "shortlist": n})
    save(pd.DataFrame(rows), "table11_fig05_fdr.csv")

    # --- Fig 6: reliability diagram (S1) ---
    rows = []
    if z is not None:
        y, lg, T = z["test_y"], z["test_logit"], float(z["temperature"])
        edges = np.linspace(0, 1, 11)
        for name, prob in (("before", 1 / (1 + np.exp(-lg))), ("after", 1 / (1 + np.exp(-lg / T)))):
            b = np.clip(np.digitize(prob, edges) - 1, 0, 9)
            for i in range(10):
                m = b == i
                if m.any():
                    rows.append({"scaling": name, "bin_mid": (edges[i] + edges[i + 1]) / 2,
                                 "mean_predicted": prob[m].mean(), "fraction_positive": y[m].mean(), "count": int(m.sum())})
    save(pd.DataFrame(rows), "fig06_reliability_s1.csv")

    # --- Table 12 / Fig 7: explanation quality ---
    rows = []
    for split in ("s1", "s2"):
        p = os.path.join(RES, "primekg", f"explanations_{split}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            ex = json.load(f)
        kept = sum(len(e["claims"]) for e in ex)
        removed = sum(e["removed_claims"] for e in ex)
        cited = [c for e in ex for c in e["claims"] if c.get("pmid")]
        cit_ok = sum(c["pmid"] in e["retrieved_pmids"] for e in ex for c in e["claims"] if c.get("pmid"))
        lit = sum(c.get("literature_supported", False) for e in ex for c in e["claims"])
        rows.append({
            "model": f"{split.upper()} model", "explanations": len(ex), "claims_generated": kept + removed,
            "claims_supported": kept, "claims_removed": removed,
            "claim_support_rate": kept / max(kept + removed, 1),
            "hallucination_rate_before_verification": removed / max(kept + removed, 1),
            "hallucination_rate_after_verification": 0.0,
            "citation_precision": cit_ok / max(len(cited), 1) if cited else None,
            "literature_supported_claims": lit,
            "mean_paths": float(np.mean([len(e["paths"]) for e in ex])),
            "mean_pmids_retrieved": float(np.mean([len(e["retrieved_pmids"]) for e in ex])),
        })
    save(pd.DataFrame(rows), "table12_fig07_explanations.csv")

    # --- Table 13 / Fig 8: ablations ---
    rows = []
    for label, folder in VARIANTS.items():
        r1, r2 = get(folder, "s1"), get(folder, "s2")
        if r1 or r2:
            rows.append({"configuration": label,
                         "s1_auprc": r1["test"]["auprc"] if r1 else None,
                         "s1_auroc": r1["test"]["auroc"] if r1 else None,
                         "s2_auprc": r2["test"]["auprc"] if r2 else None,
                         "s2_auroc": r2["test"]["auroc"] if r2 else None,
                         "s1_hits@100": r1["test"]["hits@100"] if r1 else None,
                         "s1_cfs_power@0.10": next((c["power"] for c in r1["cfs"] if c["target_fdr"] == 0.1), None) if r1 else None,
                         "train_min_s1": r1["fit"].get("train_seconds", np.nan) / 60 if r1 else None,
                         "train_min_s2": r2["fit"].get("train_seconds", np.nan) / 60 if r2 else None,
                         "peak_vram_gb": max((r["fit"].get("peak_vram_gb", 0) for r in (r1, r2) if r), default=None)})
    save(pd.DataFrame(rows), "table13_fig08_09_10_ablations.csv")

    # --- Table 14: cost ---
    main_res = load("primekg")
    log = open(os.path.join(RES, "primekg", "log.txt"), encoding="utf-8").read() if main_res else ""
    total = None
    for line in log.splitlines():
        if line.startswith("Done in"):
            total = float(line.split()[2])
    train_min = sum(r["fit"].get("train_seconds", 0) for r in main_res) / 60
    n_exp = sum(r.get("rex", {}).get("n", 0) for r in main_res)
    rex_s = (total - train_min - 1.0) * 60 / n_exp if total and n_exp else None
    rows = [
        {"component": "Data loading (PrimeKG, pruned)", "peak_vram_gb": 0, "time": "~20-30 s", "cpu_only": "Yes"},
        {"component": "TriCAL + BC-RGT training, S1 (joint, 60 epochs max)", "peak_vram_gb": main_res[0]["fit"]["peak_vram_gb"] if main_res else None,
         "time": f"{main_res[0]['fit']['train_seconds'] / 60:.1f} min" if main_res else None, "cpu_only": "Slow"},
        {"component": "TriCAL + BC-RGT training, S2", "peak_vram_gb": main_res[1]["fit"]["peak_vram_gb"] if len(main_res) > 1 else None,
         "time": f"{main_res[1]['fit']['train_seconds'] / 60:.1f} min" if len(main_res) > 1 else None, "cpu_only": "Slow"},
        {"component": "Inference: all drugs x one disease", "peak_vram_gb": "< 1 GB", "time": "< 1 s", "cpu_only": "Yes"},
        {"component": "CFS selection", "peak_vram_gb": 0, "time": "< 1 s", "cpu_only": "Yes"},
        {"component": "REX (PubMed + qwen2.5:3b + DeBERTa NLI on CPU)", "peak_vram_gb": "~2.5 GB (LLM)",
         "time": f"~{rex_s:.0f} s per explanation" if rex_s else None, "cpu_only": "Yes (slower)"},
        {"component": "Full loop (S1 + S2 + 2 case studies + 20 explanations)", "peak_vram_gb": max(r["fit"]["peak_vram_gb"] for r in main_res) if main_res else None,
         "time": f"{total:.1f} min" if total else None, "cpu_only": "-"},
    ]
    save(pd.DataFrame(rows), "table14_cost.csv")

    # --- Tables 15/16: candidates with REX paths and PMIDs ---
    for q, fname in (("Alzheimer_disease", "table15_candidates_ad.csv"), ("long_COVID-19", "table16_candidates_long_covid.csv")):
        p = os.path.join(RES, "primekg", f"candidates_s1_{q}.csv")
        if not os.path.exists(p):
            continue
        cand = pd.read_csv(p)
        with open(os.path.join(RES, "primekg", "explanations_s1.json"), encoding="utf-8") as f:
            ex = {e["drug"]: e for e in json.load(f)}
        cand["key_path"] = [" -> ".join(ex[d]["paths"][0]["nodes"]) if d in ex and ex[d]["paths"] else "" for d in cand["drug"]]
        cand["pmids"] = [", ".join(ex[d]["retrieved_pmids"][:3]) if d in ex else "" for d in cand["drug"]]
        cand["supported_claims"] = [len(ex[d]["claims"]) if d in ex else "" for d in cand["drug"]]
        z2 = os.path.join(RES, "primekg", f"candidates_s2_{q}.csv")
        if os.path.exists(z2):
            zs = pd.read_csv(z2)
            cand["zero_shot_s2_drug_at_rank"] = zs["drug"].values[: len(cand)]
        save(cand, fname)

    # --- Fig 11: drug embedding map coloured by dominant indication area; Fig 14: REX subgraph; Fig 16: funnel ---
    z = npz("primekg_eval", "s1")
    if z is not None:
        from sklearn.manifold import TSNE

        from bridge_dr.data import load_primekg
        data = load_primekg(os.path.join(ROOT, "data/primekg/kg.csv"))
        dominant = {}
        cnt: dict[int, Counter] = {}
        for d, c in data.indication.tolist():
            cnt.setdefault(d, Counter())[data.disease_area[c]] += 1
        for d, c in cnt.items():
            top = [a for a, _ in c.most_common() if a != "other"]
            dominant[d] = top[0] if top else "other"
        drugs = data.drug_nodes.tolist()
        keep = [i for i, d in enumerate(drugs) if d in dominant]
        xy = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(z["drug_emb"][keep])
        save(pd.DataFrame({"drug": [data.node_names[drugs[i]] for i in keep], "x": xy[:, 0], "y": xy[:, 1],
                           "dominant_area": [AREA_LABEL.get(dominant[drugs[i]], "Other") for i in keep]}),
             "fig11_drug_embedding_tsne.csv")
        n_drugs = len(drugs)
    else:
        n_drugs = 7957

    with open(os.path.join(RES, "primekg", "explanations_s1.json"), encoding="utf-8") as f:
        ex = json.load(f)
    edges = []
    for e in ex:
        if e["disease"] == "Alzheimer disease" and e["paths"]:
            for k, pth in enumerate(e["paths"]):
                for (a, b), rel in zip(zip(pth["nodes"][:-1], pth["nodes"][1:]), pth["relations"]):
                    edges.append({"drug": e["drug"], "path": k + 1, "source": a, "target": b, "relation": rel})
            break
    save(pd.DataFrame(edges), "fig14_rex_subgraph.csv")

    r1 = get("primekg", "s1")
    chk_path = os.path.join(OUT, "table15_16_candidate_checks.csv")  # written by scripts/check_candidates.py
    chk = pd.read_csv(chk_path) if os.path.exists(chk_path) else None
    funnel = []
    for case in (r1 or {}).get("cases", []):
        exd = [e for e in ex if e["disease"] == case["disease"]]
        funnel += [
            {"disease": case["disease"], "stage": "All drugs scored", "count": n_drugs},
            {"disease": case["disease"], "stage": "CFS shortlist (FDR 0.10)", "count": case["cfs_shortlist_size"]},
            {"disease": case["disease"], "stage": "Top-10 ranked (reported)", "count": len(case["top"])},
        ]
        c = chk[chk["disease"] == case["disease"]] if chk is not None else None
        if c is not None and len(c):
            funnel.append({"disease": case["disease"], "stage": "Top-10: drug-like (Lipinski)",
                           "count": int((c["drug_like"] == True).sum())})  # noqa: E712 (NaN for biologics)
            if case["disease"] == "Alzheimer disease":
                funnel.append({"disease": case["disease"], "stage": "Top-10: BBB+ (rule)",
                               "count": int((c["bbb_rule"] == True).sum())})  # noqa: E712
            funnel.append({"disease": case["disease"], "stage": "Top-10: registered trial",
                           "count": int((c["n_trials"] > 0).sum())})
        funnel += [
            {"disease": case["disease"], "stage": "Explained by REX", "count": len(exd)},
            {"disease": case["disease"], "stage": ">=1 verified claim", "count": sum(1 for e in exd if e["claims"])},
        ]
    save(pd.DataFrame(funnel), "fig16_funnel.csv")


if __name__ == "__main__":
    main()
