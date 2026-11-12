"""Build the supplementary document (reports/BRIDGE-DR_Supplementary.docx) from the results.

It reuses the page setup and styles of the implementation plan, and draws every number from
results/ and reports/figure_data/, so it can be regenerated after any rerun:

  python scripts/export_report_data.py && python scripts/make_figures.py && python scripts/make_supplementary.py
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import sys
from datetime import date

import docx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Emu, Pt, RGBColor  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "reports", "figure_data")
FIG = os.path.join(ROOT, "reports", "figures")
OUT = os.path.join(ROOT, "reports", "BRIDGE-DR_Supplementary.docx")
TEXT_WIDTH = Emu(5731510)
AREAS = ["neurodegenerative", "mental_health", "cardiovascular", "metabolic_disorder", "autoimmune",
         "cell_proliferation", "anemia", "adrenal_gland", "diabetes"]
BASELINE_DIRS = {"rgcn": "R-GCN", "compgcn": "CompGCN", "hgt": "HGT", "transe": "TransE",
                 "distmult": "DistMult", "rotate": "RotatE", "llm_ranking": "LLM-only ranking (qwen2.5:3b)"}
ABLATION_DIRS = {"A1_no_tricals": "A1 – no TriCAL", "A2_no_modality_dropout": "A2 – no modality dropout",
                 "A3_no_subtype_conditioning": "A3 – no subtype conditioning (FiLM)",
                 "A6_random_negatives_only": "A6 – random negatives only"}


# ---------- data access ----------
def load(folder):
    p = os.path.join(RES, folder, "results.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


def get(folder, split):
    return next((r for r in load(folder) if r["split"] == split), None)


def split_sizes(folder):
    p = os.path.join(RES, folder, "log.txt")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        m = re.match(r"\[(s\d) seed \d+\] train (\d+) val (\d+) test (\d+) contra train (\d+) calib (\d+) test-null (\d+)", line)
        if m:
            out[m.group(1)] = dict(zip(["train", "val", "test", "contra_train", "calib", "test_null"], map(int, m.groups()[1:])))
    return out


def n_params(pt):
    if not os.path.exists(pt):
        return None
    sd = torch.load(pt, map_location="cpu")
    return sum(v.numel() for v in sd.values())


def f3(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{nd}f}"


# ---------- document helpers ----------
def new_doc():
    plan = glob.glob(os.path.join(ROOT, "docs", "*Implementation_Plan*.docx"))[0]
    doc = docx.Document(plan)
    body = doc.element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"):
            body.remove(el)
    for rel in [r for r in doc.part.rels.values() if "image" in r.reltype]:
        doc.part.drop_rel(rel.rId)
    return doc


def heading(doc, text, level=1):
    style = next(st for st in doc.styles if st.type == 1 and st.name == f"Heading {level}")
    p = doc.add_paragraph(style=style)
    p.add_run(text)
    return p


def para(doc, text, bold=False, italic=False, size=11, align=None, color=None, after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic, r.font.size = bold, italic, Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style=next(st for st in doc.styles if st.type == 1 and st.name == "List Paragraph"))
        p.add_run("• " + it).font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(2)


def _cell_fmt(cell, fill=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "808080")
        borders.append(b)
    tcPr.append(borders)
    if fill:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        tcPr.append(shd)


def table(doc, caption, header, rows, widths=None, highlight_first=False, size=8.5):
    cp = doc.add_paragraph()
    r1 = cp.add_run(caption.split(" ", 2)[0] + " " + caption.split(" ", 2)[1] + " ")
    r1.bold = True
    cp.add_run(caption.split(" ", 2)[2])
    cp.paragraph_format.space_before = Pt(8)
    cp.paragraph_format.space_after = Pt(3)
    cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.autofit = widths is None
    for i, row in enumerate([header] + rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            run = c.paragraphs[0].add_run(str(v))
            run.font.size = Pt(size)
            run.bold = i == 0
            _cell_fmt(c, "D9E2F3" if i == 0 else ("FFF2CC" if highlight_first and i == 1 else None))
            if widths:
                c.width = Emu(int(TEXT_WIDTH * widths[j] / sum(widths)))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def mark_recent(t, first_col):
    """Highlight body values from the latest runs (seed-1 retraining, candidate checks) in yellow."""
    for row in t.rows[1:]:
        for c in row.cells[first_col:]:
            if c.text.strip() not in ("", "–"):
                for r in c.paragraphs[0].runs:
                    r.font.highlight_color = WD_COLOR_INDEX.YELLOW


RECENT_NOTE = "Yellow: values from the latest runs (seed-1 retraining and candidate checks)."


def figure(doc, png, caption, width=1.0):
    if not os.path.exists(png):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(png, width=Emu(int(TEXT_WIDTH * width)))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption.split(".", 1)[0] + ".")
    r.bold, r.font.size = True, Pt(9.5)
    c.add_run(caption.split(".", 1)[1]).font.size = Pt(9.5)
    c.paragraph_format.space_after = Pt(10)


# ---------- supplementary figures ----------
def training_curves_figure():
    rows = []
    for split in ("s1", "s2"):
        p = os.path.join(RES, "primekg", f"history_{split}_seed0.json")
        if os.path.exists(p):
            for h in json.load(open(p)):
                rows.append({"split": split.upper(), "epoch": h["epoch"], "train_loss": h["loss"],
                             "val_auprc": h["val_auprc"], "val_auroc": h["val_auroc"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "figS1_training_curves.csv"), index=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    colors = {"S1": "#2b7bd6", "S2": "#e8643c"}
    for s, g in df.groupby("split"):
        a.plot(g["epoch"], g["train_loss"], "-", color=colors[s], lw=2, label=s)
        b.plot(g["epoch"], g["val_auprc"], "-", color=colors[s], lw=2, label=f"{s} AUPRC")
        b.plot(g["epoch"], g["val_auroc"], "--", color=colors[s], lw=1.5, label=f"{s} AUROC")
    a.set_yscale("log")
    a.set_xlabel("Epoch")
    a.set_ylabel("Training loss (log)")
    b.set_xlabel("Epoch")
    b.set_ylabel("Validation score")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG, "figS1_training_curves.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


# ---------- main ----------
def main():
    doc = new_doc()
    main1, main2 = get("primekg", "s1"), get("primekg", "s2")
    sizes = split_sizes("primekg")

    para(doc, "SUPPLEMENTARY MATERIAL", bold=True, size=16, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    para(doc, "BRIDGE-DR: A Biomarker-Conditioned Multimodal Knowledge-Graph Transformer with Conformal "
              "False-Discovery Control and Retrieval-Grounded LLM Explanations for Patient-Stratified Drug "
              "Repurposing in Alzheimer's Disease and Long COVID", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, f"Implementation run on PrimeKG · 1 seed · generated {date.today():%d %B %Y}", italic=True, size=10,
         align=WD_ALIGN_PARAGRAPH.CENTER, color="444444", after=12)

    heading(doc, "Contents", 1)
    bullets(doc, ["S1. Software and hardware environment", "S2. Data and preprocessing statistics",
                  "S3. Hyperparameters", "S4. Model sizes", "S5. Training curves",
                  "S6. Zero-shot results for every disease area", "S7. Comparison with baseline methods",
                  "S8. Ablation results", "S9. Calibration and conformal FDR control",
                  "S10. Candidate drug lists", "S11. REX explanations", "S12. Reproducibility",
                  "S13. Deviations from the implementation plan and limitations"])

    # S1
    heading(doc, "S1. Software and hardware environment", 1)
    import sklearn

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "not installed"

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB" if torch.cuda.is_available() else "–"
    table(doc, "Table S1. Environment used for all runs", ["Component", "Version / specification"], [
        ["Operating system", f"{platform.system()} {platform.release()}"],
        ["GPU", f"{gpu} ({vram})"],
        ["Python", platform.python_version()],
        ["PyTorch / CUDA", f"{torch.__version__} / {torch.version.cuda}"],
        ["NumPy / pandas / scikit-learn", f"{np.__version__} / {pd.__version__} / {sklearn.__version__}"],
        ["RDKit", ver("rdkit")],
        ["transformers (NLI verifier)", ver("transformers")],
        ["Local LLM (REX)", "qwen2.5:3b served by Ollama"],
        ["NLI model", "cross-encoder/nli-deberta-v3-small"],
    ], widths=[1, 2])

    # S2
    heading(doc, "S2. Data and preprocessing statistics", 1)
    para(doc, "PrimeKG (kg.csv, Harvard Dataverse file 6180620) was loaded once and pruned so that the graph and "
              "the model fit in 16 GB RAM and 4 GB VRAM: every structural relation was capped at 150,000 edges by "
              "seeded random sampling. Relations are direction-aware (e.g. drug_protein:drug→gene/protein and its "
              "inverse are separate types). Label relations (indication, off-label use, contraindication) are kept "
              "out of the structural graph and added back per split using training pairs only.", size=10.5)
    try:
        from bridge_dr.data import load_primekg
        data = load_primekg(os.path.join(ROOT, "data/primekg/kg.csv"))
        cnt = pd.Series(data.node_type.numpy()).value_counts()
        table(doc, "Table S2. Node types", ["Node type", "Count"],
              [[data.node_type_names[i], f"{int(c):,}"] for i, c in cnt.items()], widths=[2, 1])
        et = pd.Series(data.edge_type.numpy()).value_counts().sort_index()
        table(doc, "Table S3. Structural relations after pruning (cap 150,000 per relation)", ["Relation (direction)", "Edges"],
              [[data.relation_names[i], f"{int(c):,}"] for i, c in et.items()] +
              [["Total", f"{int(et.sum()):,}"]], widths=[3, 1], size=7.5)
        table(doc, "Table S4. Labels and drug features", ["Item", "Count"], [
            ["Drugs / diseases", f"{len(data.drug_nodes):,} / {len(data.disease_nodes):,}"],
            ["Indication + off-label pairs (positives)", f"{len(data.indication):,}"],
            ["Contraindication pairs (confirmed negatives)", f"{len(data.contraindication):,}"],
            ["Drugs with Morgan fingerprint (PubChem SMILES)", f"{int(data.drug_mol_mask.sum()):,}"],
            ["Drugs with L1000 signature", f"{int(data.drug_tx_mask.sum()):,} (L1000 not supplied)"],
            ["Disease signature genes (KG-derived)", f"{len(data.sig_genes)}"],
        ], widths=[3, 1])
    except Exception as e:
        para(doc, f"(PrimeKG statistics unavailable: {e})", italic=True, size=9)
    rows = []
    for name, folder in [("Main (S1)", "primekg"), ("Main (S2, neurodegenerative)", "primekg")] + \
            [(f"S2 – {a.replace('_', ' ')}", f"areas/{a}") for a in AREAS[1:]]:
        s = split_sizes(folder).get("s1" if "S1" in name else "s2")
        if s:
            rows.append([name] + [f"{s[k]:,}" for k in ("train", "val", "test", "contra_train", "calib", "test_null")])
    table(doc, "Table S5. Split sizes (pairs)", ["Experiment", "Train +", "Val +", "Test +", "Contra (train)",
                                                 "CFS calibration", "Test nulls"], rows, widths=[3, 1, 1, 1, 1.2, 1.2, 1])

    # S3
    heading(doc, "S3. Hyperparameters", 1)
    from bridge_dr.trainer import TrainConfig
    cfg = TrainConfig()
    table(doc, "Table S6. Training and model hyperparameters (identical for all experiments unless stated)",
          ["Parameter", "Value"], [
              ["Hidden size / layers / heads", f"{cfg.dim} / {cfg.layers} / {cfg.heads}"],
              ["Dropout", cfg.dropout],
              ["Optimiser", f"AdamW, lr {cfg.lr}, weight decay {cfg.weight_decay}, cosine schedule, grad-clip 1.0"],
              ["Epochs / early stopping", f"≤ {cfg.epochs}, patience {cfg.patience} evaluations on validation AUPRC "
                                          f"(evaluated every {cfg.eval_every} epochs)"],
              ["Batch size (positive pairs)", cfg.batch_size],
              ["Edges sampled per training step / at evaluation", f"{cfg.max_edges:,} / {cfg.eval_max_edges:,}"],
              ["Negatives per positive", "1 random drug (known positives filtered) + 1 hard (training contraindication)"],
              ["Loss", f"focal (γ = 2, α = 0.5) on indications + focal on contraindications + {cfg.w_tricals}·TriCAL "
                       f"(+ {cfg.w_srcl}·SRCL with margin {cfg.srcl_margin} when L1000 is present)"],
              ["TriCAL", f"InfoNCE τ = 0.07, modality dropout {cfg.modality_dropout}, gated fusion"],
              ["Score scaling", "decoder logits divided by √dim"],
              ["CFS", "α ∈ {0.05, 0.10, 0.20}; per-area calibration, pooled if < 30 negatives"],
              ["REX", "≤ 3-hop paths, top-5; 10 PubMed abstracts (BM25 over ≤ 40 hits); NLI entailment > 0.5"],
              ["Random seed", "0"],
          ], widths=[1.3, 2.7])

    # S4
    heading(doc, "S4. Model sizes", 1)
    rows = [["BRIDGE-DR (TriCAL + BC-RGT)", f"{n_params(os.path.join(ROOT, 'checkpoints', 'model_s1_seed0.pt')) or 0:,}"]]
    for d, name in BASELINE_DIRS.items():
        n = n_params(os.path.join(RES, "baselines", d, "model_s1_seed0.pt"))
        if n:
            rows.append([name, f"{n:,}"])
    table(doc, "Table S7. Trainable parameters (most are the 129,375 × 128 node-embedding table)", ["Model", "Parameters"],
          rows, widths=[3, 1])

    # S5
    heading(doc, "S5. Training curves", 1)
    png = training_curves_figure()
    if png:
        figure(doc, png, "Figure S1. Training loss and validation AUROC/AUPRC per epoch for the main S1 and S2 runs.")

    # S6
    heading(doc, "S6. Zero-shot results for every disease area", 1)
    para(doc, "For each area, all indications of diseases in that area were removed from training (and their "
              "contraindications from training and calibration), and the model was trained from scratch.", size=10.5)
    rows = []
    for a in AREAS:
        r = main2 if a == "neurodegenerative" else get(f"areas/{a}", "s2")
        if r:
            t, f = r["test"], r["fit"]
            rows.append([a.replace("_", " ").capitalize(), f3(t["auroc"]), f3(t["auprc"]), f3(t["mrr"]), f3(t["hits@10"]),
                         f3(t["hits@100"]), f"{t['median_rank']:.0f}", f3(f.get("train_seconds", np.nan) / 60, 1)])
    table(doc, "Table S8. S2 zero-shot results per held-out disease area (BRIDGE-DR, measured)",
          ["Area", "AUROC", "AUPRC", "MRR", "Hits@10", "Hits@100", "Median rank", "Train (min)"], rows,
          widths=[2.2, 1, 1, 1, 1, 1, 1.1, 1])

    # S7
    heading(doc, "S7. Comparison with baseline methods", 1)
    para(doc, "Baselines were reimplemented in the same framework and trained on the same graph, splits, negatives "
              "and metrics (1 seed). GNN baselines replace the BC-RGT layer and use the same DistMult decoder, without "
              "TriCAL or FiLM conditioning; KG embeddings are trained on all graph triples plus training labels. "
              "The LLM-only baseline scores a sample of the same test pairs with a local 3B model, without the graph. "
              "TxGNN, NBFNet and CMap reversal were not run and are not reported in the main report.",
         size=10.5)
    rows = []
    for split, label in (("s1", "S1"), ("s2", "S2 (neurodeg.)")):
        r = main1 if split == "s1" else main2
        if r:
            t = r["test"]
            rows.append([f"BRIDGE-DR", label, f3(t["auroc"]), f3(t["auprc"]), f3(t["hits@10"]), f3(t["mrr"]),
                         f3(r["fit"].get("train_seconds", np.nan) / 60, 1)])
        for d, name in BASELINE_DIRS.items():
            b = get(f"baselines/{d}", split)
            if b:
                t = b["test"]
                rows.append([name, label, f3(t["auroc"]), f3(t["auprc"]), f3(t.get("hits@10")), f3(t.get("mrr")),
                             f3(b["fit"].get("train_seconds", np.nan) / 60, 1) if "llm" not in b else
                             f"{b['fit'].get('inference_seconds', 0) / 60:.1f} (inference)"])
    table(doc, "Table S9. BRIDGE-DR vs measured baselines", ["Method", "Split", "AUROC", "AUPRC", "Hits@10", "MRR",
                                                            "Train (min)"], rows, widths=[2.4, 1.3, 1, 1, 1, 1, 1.1])
    sc_p = os.path.join(ROOT, "reports", "figure_data", "seed_check_s2.csv")
    if os.path.exists(sc_p):
        sc = pd.read_csv(sc_p)
        para(doc, "Seed check. TransE scored higher than BRIDGE-DR on S2 with seed 0, so both were retrained with a second "
                  "seed on the same split (the split itself is fixed by the held-out area; the seed changes initialisation, "
                  "negative sampling and edge sampling).", size=10.5)
        rows = [[m, *(f3(v) for v in g["auprc"]), f"{g['auprc'].mean():.3f} ± {g['auprc'].std():.3f}",
                 f"{g['auroc'].mean():.3f} ± {g['auroc'].std():.3f}"] for m, g in sc.groupby("method", sort=False)]
        para(doc, RECENT_NOTE, italic=True, size=9)
        sb = table(doc, "Table S9b. S2 (neurodegenerative) over two seeds", ["Method", "AUPRC seed 0", "AUPRC seed 1",
                                                                         "AUPRC mean ± SD", "AUROC mean ± SD"],
              rows, widths=[1.6, 1.1, 1.1, 1.4, 1.4])
        mark_recent(sb, 1)

    # S8
    heading(doc, "S8. Ablation results", 1)
    rows = []
    for d, name in [("primekg", "BRIDGE-DR (full)")] + [(f"ablations/{k}", v) for k, v in ABLATION_DIRS.items()]:
        r1, r2 = get(d, "s1"), get(d, "s2")
        if r1 or r2:
            rows.append([name, f3(r1["test"]["auroc"]) if r1 else "–", f3(r1["test"]["auprc"]) if r1 else "–",
                         f3(r1["test"]["hits@100"]) if r1 else "–", f3(r2["test"]["auroc"]) if r2 else "–",
                         f3(r2["test"]["auprc"]) if r2 else "–"])
    table(doc, "Table S10. Ablations (S2 = neurodegenerative held out)",
          ["Configuration", "S1 AUROC", "S1 AUPRC", "S1 Hits@100", "S2 AUROC", "S2 AUPRC"], rows,
          widths=[2.6, 1, 1, 1, 1, 1], highlight_first=True)
    para(doc, "A4 (no SRCL) equals the full model because SRCL needs L1000 data, which was not supplied. A5 (unweighted "
              "literature edges) does not apply: this build adds no literature co-mention edges. A7 (top-k instead of "
              "CFS) is reported in S9. A8/A9 (REX without retrieval / verification) are summarised in S11.", italic=True,
         size=9.5, color="444444")

    # S9
    heading(doc, "S9. Calibration and conformal FDR control", 1)
    rows = []
    for r in (main1, main2):
        if not r:
            continue
        for c in r["cfs"]:
            rows.append([r["split"].upper(), "CFS", c["target_fdr"], f3(c["empirical_fdr"]), f3(c["power"]), c["shortlist_size"]])
        t = r["cfs_topk_baseline"]
        rows.append([r["split"].upper(), "Top-k (same size as CFS at 0.10)", "none", f3(t["empirical_fdr"]), f3(t["power"]),
                     t["shortlist_size"]])
    table(doc, "Table S11. Conformal FDR-controlled selection", ["Split", "Rule", "Target FDR", "Empirical FDR", "Power",
                                                                 "Shortlist"], rows, widths=[0.7, 2.6, 1, 1, 1, 1])
    rows = []
    for r in (main1, main2):
        if r:
            c = r["calibration"]
            rows.append([r["split"].upper(), f3(c["temperature"]), f3(c["before"]["ece"]), f3(c["after"]["ece"]),
                         f3(c["before"]["brier"]), f3(c["after"]["brier"])])
    table(doc, "Table S12. Temperature scaling (fitted on validation)", ["Split", "Temperature", "ECE before", "ECE after",
                                                                         "Brier before", "Brier after"], rows)
    figure(doc, os.path.join(FIG, "fig05_fdr_control.png"), "Figure S2. Empirical vs target FDR (S1).", 0.6)

    # S10
    heading(doc, "S10. Candidate drug lists", 1)
    para(doc, "Top-10 drugs per case disease, excluding known training/validation indications. p is the conformal p-value "
              "against held-out contraindications of the same disease area. No candidate passed the per-disease "
              "Benjamini–Hochberg shortlist at FDR 0.10 over all 7,957 drugs.", size=10.5)
    n = 13
    for split, model in (("s1", "S1 model (all disease areas seen)"), ("s2", "S2 model (neurodegenerative indications never seen)")):
        for q in ("Alzheimer_disease", "long_COVID-19"):
            p = os.path.join(RES, "primekg", f"candidates_{split}_{q}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                table(doc, f"Table S{n}. {q.replace('_', ' ')} – {model}", ["Rank", "Drug", "Score", "Conformal p",
                                                                             "Known test positive"],
                      [[int(r["rank"]), r["drug"], f3(r["score"]), f"{r['conformal_p']:.4f}",
                        "yes" if r["known_test_positive"] else "no"] for _, r in df.iterrows()],
                      widths=[0.6, 2.4, 1, 1, 1.2])
                n += 1
    chk_p = os.path.join(ROOT, "reports", "figure_data", "table15_16_candidate_checks.csv")
    if os.path.exists(chk_p):
        chk = pd.read_csv(chk_p)
        para(doc, "Candidate checks (S1 top-10, scripts/check_candidates.py). Drug-likeness and BBB are rules of thumb on the "
                  "PubChem structure, not a trained ADMET model: Lipinski (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10; drug-like "
                  "with ≤ 1 violation) and BBB+ (TPSA < 90 Å², MW < 450, HBD ≤ 3). Trials are ClinicalTrials.gov studies "
                  "listing the drug as an intervention for the condition (long COVID: Post-Acute COVID-19 Syndrome); a "
                  "listed trial may use the drug as a comparator.", size=10.5)

        def yn(v):
            return "–" if pd.isna(v) else ("yes" if str(v) == "True" else "no")
        para(doc, RECENT_NOTE, italic=True, size=9)
        sc_t = table(doc, f"Table S{n}. Drug-likeness, BBB and trial status of the top-10 candidates",
              ["Disease", "Rank", "Drug", "MW", "TPSA", "Lipinski viol.", "Drug-like", "BBB+", "Trials"],
              [[r["disease"], int(r["rank"]), r["drug"],
                "biologic" if not r["has_structure"] else f"{r['mw']:.0f}",
                "–" if pd.isna(r["tpsa"]) else f"{r['tpsa']:.0f}",
                "–" if pd.isna(r["lipinski_violations"]) else int(r["lipinski_violations"]),
                yn(r["drug_like"]), yn(r["bbb_rule"]), r["trial_status"]] for _, r in chk.iterrows()],
              widths=[1.1, 0.4, 1.4, 0.5, 0.5, 0.6, 0.6, 0.5, 1.4], size=8)
        mark_recent(sc_t, 3)
        n += 1

    # S11
    heading(doc, "S11. REX explanations", 1)
    para(doc, "For each explained prediction: the top KG paths (attention × gradient), the PubMed abstracts retrieved, "
              "and the claims that survived NLI verification against their cited abstract. Removed claims are counted. "
              "A claim generated without any retrieved abstract (template generator) is supported by its KG path only.",
         size=10.5)
    for split in ("s1", "s2"):
        p = os.path.join(RES, "primekg", f"explanations_{split}.json")
        if not os.path.exists(p):
            continue
        heading(doc, f"{split.upper()} model", 2)
        for e in json.load(open(p, encoding="utf-8")):
            para(doc, f"{e['drug']} → {e['disease']}", bold=True, size=10.5, after=2)
            lines = [" → ".join(pth["nodes"]) for pth in e["paths"][:3]] or ["no path found"]
            bullets(doc, ["Paths: " + "; ".join(lines),
                          f"PMIDs retrieved: {', '.join(e['retrieved_pmids']) or 'none'}",
                          f"Generator: {e['generator']}; supported claims: {len(e['claims'])}; removed: {e['removed_claims']}"] +
                    [f"Claim: {c['claim']} [PMID {c.get('pmid') or '–'}]" for c in e["claims"]])

    # S12
    heading(doc, "S12. Reproducibility", 1)
    para(doc, "All code, trained checkpoints and raw outputs are in the project repository. The commands below "
              "reproduce every number in the report and this supplement.", size=10.5)
    for cmd in ["pip install -r requirements.txt",
                "python scripts/fetch_smiles.py --out data/primekg/drug_smiles.csv",
                "python scripts/run_experiments.py",
                "python scripts/export_report_data.py && python scripts/make_figures.py",
                "python scripts/fill_report.py && python scripts/make_supplementary.py"]:
        p = para(doc, cmd, size=9, after=2)
        p.runs[0].font.name = "Consolas"
    bullets(doc, ["Seed 0 for data pruning, splits, initialisation and sampling.",
                  "Trained models: checkpoints/model_s1_seed0.pt and checkpoints/model_s2_seed0.pt; reuse with "
                  "--resume --checkpoint-dir checkpoints (no retraining).",
                  "Raw outputs per experiment: results/<experiment>/results.json, summary.md, log.txt.",
                  "Every figure is drawn from an editable CSV in reports/figure_data/.",
                  "See USER_GUIDE.md for all options."])

    # S13
    heading(doc, "S13. Deviations from the implementation plan and limitations", 1)
    bullets(doc, [
        "Scale: hidden 128 and 2 layers instead of 256 and 3; random edge sampling (300k per step) instead of neighbour "
        "sampling; 1 seed instead of 5; no Optuna tuning.",
        "Drug structure: Morgan fingerprints only (no ChemBERTa embedding); 228 labelled drugs without a PubChem SMILES "
        "use the graph view only.",
        "Transcriptome: LINCS L1000 not used, so TriCAL fuses two views and SRCL is inactive (ablation A4 = full model).",
        "Disease signatures: derived from KG gene neighbourhoods (the plan's fallback); GEO normalisation, limma, ComBat "
        "and NMF subtypes not run, so Table 10 of the plan is not filled.",
        "Disease areas for S2 are defined by name keywords and approximate TxGNN's ontology-based areas.",
        "S3 (time-sliced) not run: AACT trial start dates were not supplied.",
        "Baselines are lightweight reimplementations, not the official code; TxGNN, NBFNet and CMap reversal were not run.",
        "REX: BM25 retrieval only (no MedCPT/FAISS dense index); 3B instead of 7–8B LLM; most raw claims fail NLI "
        "verification and are removed. NLI checks a claim only against its cited abstract, not whether the claim is "
        "relevant to the predicted indication, so some verified claims are true but off-topic (e.g. Filgrastim and "
        "chemotherapy-induced neutropenia for long COVID). Expert scoring not done.",
        "Validation stage: drug-likeness and BBB are rule-based screens and trial status is a ClinicalTrials.gov count; CMap permutation test, docking, trained ADMET models and CTD cross-check not run.",
        "Per-disease CFS applies Benjamini–Hochberg over all 7,957 drugs; no single-disease shortlist passed FDR 0.10.",
    ])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        doc.save(OUT)
        print(f"Saved {OUT}")
    except PermissionError:
        alt = OUT.replace(".docx", "_updated.docx")
        doc.save(alt)
        print(f"{OUT} is open in another program; saved {alt} instead")


if __name__ == "__main__":
    main()
