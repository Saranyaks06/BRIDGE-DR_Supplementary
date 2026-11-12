"""Fill the implementation plan document with the results of this implementation.

Reads docs/<plan>.docx (never changed), the CSVs in reports/figure_data/ and the PNGs in
reports/figures/, and writes reports/BRIDGE-DR_Implementation_Report.docx.

Sections 1-3 (objective, datasets, methodology) are kept as written in the plan. Section 4 holds
only results computed by this implementation: rows, columns, tables and sections that were not
run are removed, and sections, tables and figures are renumbered. Values from the implementation
are highlighted yellow; headings and captions of implemented parts green. A literature review,
references and an appendix of further results are added at the end.

  python scripts/export_report_data.py && python scripts/make_figures.py && python scripts/fill_report.py
"""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import sys

import docx
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reports", "figure_data")
FIG = os.path.join(ROOT, "reports", "figures")
OUT = os.path.join(ROOT, "reports", "BRIDGE-DR_Implementation_Report.docx")
WITH_LITERATURE = "--with-literature" in sys.argv  # section 5, references, appendix heading and scope note
VALUE_MARK = WD_COLOR_INDEX.YELLOW        # value produced by this implementation
TITLE_MARK = WD_COLOR_INDEX.BRIGHT_GREEN  # heading / caption of a part that was implemented and run


def csv(name):
    p = os.path.join(DATA, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


# ---------- low-level editing helpers ----------
def set_cell(cell, text, bold=None):
    p = cell.paragraphs[0]
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    runs = p.runs
    if not runs:
        r = p.add_run(str(text))
        r.font.size = docx.shared.Pt(9)
    else:
        runs[0].text = str(text)
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
        r = runs[0]
    if bold is not None:
        r.bold = bold


def set_row(row, values, bold=None):
    for c, v in zip(row.cells, values):
        set_cell(c, v, bold)


def set_par(par, *parts):
    """parts: (text, bold) pairs; reuses the paragraph's first run formatting."""
    runs = par.runs
    tmpl = copy.deepcopy(runs[0]._r) if runs else None
    for r in runs:
        r._r.getparent().remove(r._r)
    for text, bold in parts:
        if tmpl is not None:
            el = copy.deepcopy(tmpl)
            par._p.append(el)
            run = docx.text.run.Run(el, par)
            run.text = text
        else:
            run = par.add_run(text)
        run.bold = bold


def insert_after(anchor_el, template_par, *parts):
    el = copy.deepcopy(template_par._p)
    anchor_el.addnext(el)
    p = docx.text.paragraph.Paragraph(el, template_par._parent)
    set_par(p, *parts)
    return p


def picture_size(png, width_emu):
    w, h = Image.open(png).size
    return width_emu, int(width_emu * h / w)


def replace_image(doc, par, png):
    blip = par._p.find(".//" + qn("a:blip"))
    rid, _ = doc.part.get_or_add_image(png)
    blip.set(qn("r:embed"), rid)
    ext = par._p.find(".//" + qn("wp:extent"))
    cx, cy = picture_size(png, int(ext.get("cx")))
    for e in (ext, par._p.find(".//" + qn("a:ext"))):
        e.set("cx", str(cx))
        e.set("cy", str(cy))


def insert_figure_after(doc, anchor_el, fig_par_template, png, width_emu=5200000):
    el = copy.deepcopy(fig_par_template._p)
    anchor_el.addnext(el)
    p = docx.text.paragraph.Paragraph(el, fig_par_template._parent)
    ext = el.find(".//" + qn("wp:extent"))
    ext.set("cx", str(width_emu))
    el.find(".//" + qn("a:ext")).set("cx", str(width_emu))
    replace_image(doc, p, png)
    return p


def resize_rows(table, n_data):
    """Keep the header, make exactly n_data body rows by copying the first body row."""
    tbl = table._tbl
    body = table.rows[1:]
    tmpl = copy.deepcopy(body[0]._tr)
    for r in body:
        tbl.remove(r._tr)
    for _ in range(n_data):
        tbl.append(copy.deepcopy(tmpl))


def find_par(doc, startswith):
    return next(p for p in doc.paragraphs if p.text.strip().startswith(startswith))


# ---------- appendix helpers ----------
def add_copy(doc, template_par, *parts):
    """Append a paragraph at the end of the body that copies template_par's formatting."""
    p = doc.add_paragraph()
    el = copy.deepcopy(template_par._p)
    p._p.addnext(el)
    p._p.getparent().remove(p._p)
    q = docx.text.paragraph.Paragraph(el, template_par._parent)
    for pr in el.findall(qn("w:pPr")):  # no list numbering in copies
        num = pr.find(qn("w:numPr"))
        if num is not None:
            pr.remove(num)
    set_par(q, *parts)
    return q


def add_table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "808080")
        borders.append(b)
    t._tbl.tblPr.append(borders)
    for i, row in enumerate([header] + rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            r = c.paragraphs[0].add_run(str(v))
            r.font.size = docx.shared.Pt(9)
            r.bold = i == 0
            if i == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:fill"), "D9E2F3")
                c._tc.get_or_add_tcPr().append(shd)
            if widths:
                c.width = docx.shared.Inches(widths[j])
    doc.add_paragraph()
    return t


def add_figure(doc, png, width_in=6.0):
    if not os.path.exists(png):
        return None
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(png, width=docx.shared.Inches(width_in))
    return p


def f(x, nd=3):
    return f"{float(x):.{nd}f}"


def measured(src):
    return str(src).strip().lower().startswith("measured")


def drop(el):
    el.getparent().remove(el)


def fill(cell, text):
    """Replace a cell's text with a value from this implementation, highlighted."""
    set_cell(cell, text)
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.highlight_color = VALUE_MARK


def mark(par, color):
    for r in par.runs:
        r.font.highlight_color = color


def caption(par, label, text):
    set_par(par, (label, True), (" " + text, False))
    mark(par, TITLE_MARK)


def heading(par, text):
    set_par(par, (text, False))
    mark(par, TITLE_MARK)


def col_index(table, header_startswith):
    return next(i for i, c in enumerate(table.rows[0].cells) if c.text.strip().startswith(header_startswith))


def delete_rows(table, labels):
    for row in list(table.rows[1:]):
        if any(row.cells[0].text.strip().startswith(lb) for lb in labels):
            drop(row._tr)


def delete_cols(table, idxs):
    for tr in table._tbl.findall(qn("w:tr")):
        tcs = tr.findall(qn("w:tc"))
        for i in sorted(idxs, reverse=True):
            if i < len(tcs):
                drop(tcs[i])
    cols = table._tbl.tblGrid.findall(qn("w:gridCol"))
    for i in sorted(idxs, reverse=True):
        if i < len(cols):
            drop(cols[i])


def plain_paragraph(p):
    for pr in p._p.findall(qn("w:pPr")):
        num = pr.find(qn("w:numPr"))
        if num is not None:
            pr.remove(num)


AREA_ORDER = ["Neurodegenerative", "Mental health", "Cardiovascular", "Metabolic disorder", "Autoimmune",
              "Cell proliferation", "Anaemia", "Adrenal gland", "Diabetes"]  # = row order of Table 8

REFERENCES = [
    "Pushpakom S, Iorio F, Eyers PA, et al. Drug repurposing: progress, challenges and recommendations. "
    "Nat Rev Drug Discov. 2019;18(1):41–58. doi:10.1038/nrd.2018.168",
    "Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine. Sci Data. "
    "2023;10:67. doi:10.1038/s41597-023-01960-3",
    "Himmelstein DS, Lizee A, Hessler C, et al. Systematic integration of biomedical knowledge prioritizes drugs "
    "for repurposing. eLife. 2017;6:e26726. doi:10.7554/eLife.26726",
    "Huang K, Chandak P, Wang Q, et al. A foundation model for clinician-centered drug repurposing. Nat Med. "
    "2024;30(12):3601–3613. doi:10.1038/s41591-024-03233-x",
    "Bordes A, Usunier N, Garcia-Duran A, et al. Translating embeddings for modeling multi-relational data. "
    "Advances in Neural Information Processing Systems 26 (NIPS 2013).",
    "Yang B, Yih W, He X, et al. Embedding entities and relations for learning and inference in knowledge bases. "
    "ICLR 2015. arXiv:1412.6575",
    "Sun Z, Deng ZH, Nie JY, Tang J. RotatE: knowledge graph embedding by relational rotation in complex space. "
    "ICLR 2019. arXiv:1902.10197",
    "Schlichtkrull M, Kipf TN, Bloem P, et al. Modeling relational data with graph convolutional networks. "
    "ESWC 2018. arXiv:1703.06103",
    "Vashishth S, Sanyal S, Nitin V, Talukdar P. Composition-based multi-relational graph convolutional networks. "
    "ICLR 2020. arXiv:1911.03082",
    "Hu Z, Dong Y, Wang K, Sun Y. Heterogeneous graph transformer. The Web Conference (WWW) 2020. arXiv:2003.01332",
    "Zhu Z, Zhang Z, Xhonneux LP, Tang J. Neural Bellman-Ford networks: a general graph neural network framework "
    "for link prediction. NeurIPS 2021. arXiv:2106.06935",
    "Lamb J, Crawford ED, Peck D, et al. The Connectivity Map: using gene-expression signatures to connect small "
    "molecules, genes, and disease. Science. 2006;313(5795):1929–1935. doi:10.1126/science.1132939",
    "Subramanian A, Narayan R, Corsello SM, et al. A next generation Connectivity Map: L1000 platform and the first "
    "1,000,000 profiles. Cell. 2017;171(6):1437–1452. doi:10.1016/j.cell.2017.10.049",
    "Jin Y, Candès EJ. Selection by prediction with conformal p-values. J Mach Learn Res. 2023. arXiv:2210.01408",
    "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple "
    "testing. J R Stat Soc B. 1995;57(1):289–300. doi:10.1111/j.2517-6161.1995.tb02031.x",
    "Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017. arXiv:1706.04599",
    "Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. "
    "NeurIPS 2020. arXiv:2005.11401",
    "Saranya K, Joseph ER, Kalaiarasi T, Karthiga M. Generative AI in drug repurposing and biomarker discovery: "
    "a multimodal approach. Front Bioinform. 2026;6:1755412. doi:10.3389/fbinf.2026.1755412",
    "Liao Y, Shen C, Sun L, et al. MRDGNN: a multi-relational reasoning framework for predicting drug indications "
    "via relational digraphs. Comput Biol Chem. 2026;123:109065. doi:10.1016/j.compbiolchem.2026.109065",
]


# ---------- main ----------
def main():
    plan = glob.glob(os.path.join(ROOT, "docs", "*Implementation_Plan*.docx"))[0]
    doc = docx.Document(plan)
    T = doc.tables
    P = {k: find_par(doc, k) for k in (
        "4. EXPECTED RESULTS", "All values and graphs in this section", "4.1 ", "4.2 ", "4.3 ", "4.4 ", "4.5 ",
        "4.6 ", "4.7 ", "4.8 ", "4.9 ", "4.10 ", "Table 7.", "Table 8.", "Table 9.", "Table 10.", "Table 11.",
        "Table 12.", "Table 13.", "Table 14.", "Table 15.", "Table 16.", "Figure 2.", "Figure 3.", "Figure 4.",
        "Figure 5.", "Figure 6.", "Figure 7.", "Figure 8.", "Figure 9.", "Figure 10.", "Subtype names are",
        "* Random negatives", "Same columns as Table 15", "This work proposes BRIDGE-DR")}
    img = {k: next(p for p in doc.paragraphs if f"rId{k}" in p._p.xml) for k in range(9, 18)}
    fig_tmpl, cap_tmpl, tab_cap_tmpl = img[9], P["Figure 2."], P["Table 7."]
    head1, head2, body_tmpl = P["4. EXPECTED RESULTS"], P["4.1 "], P["This work proposes BRIDGE-DR"]

    t7, t11, t12 = csv("table07_s1_prediction.csv"), csv("table11_fig05_fdr.csv"), csv("table12_fig07_explanations.csv")
    t13, t14 = csv("table13_fig08_09_10_ablations.csv"), csv("table14_cost.csv")
    t15, t16 = csv("table15_candidates_ad.csv"), csv("table16_candidates_long_covid.csv")
    b1, areas, rgcn = csv("baselines_s1.csv"), csv("appendix_areas.csv"), csv("rgcn_zero_shot_areas.csv")
    full = t7.iloc[0]

    # ---- Section 4 heading and note ----
    heading(P["4. EXPECTED RESULTS"], "4. RESULTS")
    set_par(P["All values and graphs in this section"], (
        "All values and graphs in this section were computed by the implementation described in Appendix A.1 "
        "(PrimeKG, laptop RTX 3050 4 GB, seed 0 unless stated). Values from the implementation are highlighted "
        "yellow; figures are drawn from the editable CSVs in reports/figure_data/.", False))
    for k in ("4.1 ", "4.2 "):
        mark(P[k], TITLE_MARK)

    # ---- Table 7: S1 prediction, measured methods only ----
    caption(P["Table 7."], "Table 7.", "Transductive link prediction on PrimeKG (S1, seed 0)")
    delete_rows(T[6], ("TxGNN", "NBFNet", "CMap", "LLM ranking"))
    for c, k in zip(T[6].rows[1].cells[1:], ("auroc", "auprc", "hits@10", "mrr")):
        fill(c, f(full[k]))
    for row in T[6].rows[2:]:
        b = b1[(b1["method"] == row.cells[0].text.strip()) & b1["source"].map(measured)].iloc[0]
        for c, k in zip(row.cells[1:], ("auroc", "auprc", "hits@10", "mrr")):
            fill(c, f(b[k]))
    replace_image(doc, img[9], os.path.join(FIG, "fig02_pr_curve_s1.png"))
    caption(P["Figure 2."], "Figure 2.", "(a) Precision–recall curves on PrimeKG (S1) for the full model and the "
                                         "ablations; (b) S1 AUPRC of BRIDGE-DR and the reimplemented baselines "
                                         "(hatched bars: the plan's values for methods not run here).")

    # ---- Table 8: zero-shot per area, BRIDGE-DR and R-GCN ----
    caption(P["Table 8."], "Table 8.", "Zero-shot AUPRC per held-out disease area (S2, seed 0)")
    rg = dict(zip(rgcn["area"], rgcn["auprc"])) if len(rgcn) else {}
    br = dict(zip(areas["area"], areas["auprc"]))
    have_rgcn = all(a in rg for a in AREA_ORDER)
    hdr = [c.text.strip() for c in T[7].rows[0].cells]
    delete_cols(T[7], [hdr.index(m) for m in ("TxGNN", "NBFNet", "HGT") + (() if have_rgcn else ("R-GCN",))])
    for row, a in zip(T[7].rows[1:10], AREA_ORDER):
        fill(row.cells[1], f(br[a]))
        if have_rgcn:
            fill(row.cells[2], f(rg[a]))
    sd = lambda v: f"{pd.Series(v).mean():.3f} ± {pd.Series(v).std():.3f}"  # noqa: E731
    fill(T[7].rows[10].cells[1], sd([br[a] for a in AREA_ORDER]))
    if have_rgcn:
        fill(T[7].rows[10].cells[2], sd([rg[a] for a in AREA_ORDER]))
    replace_image(doc, img[10], os.path.join(FIG, "fig03_zero_shot_heatmap.png"))
    caption(P["Figure 3."], "Figure 3.", "Zero-shot AUPRC per held-out disease area (S2) for BRIDGE-DR and R-GCN, "
                                         "the methods run on all nine areas.")
    replace_image(doc, img[11], os.path.join(FIG, "fig04_recall_at_k.png"))
    caption(P["Figure 4."], "Figure 4.", "Recall@k of BRIDGE-DR on the random (S1) and zero-shot (S2) splits.")

    # ---- remove the sections that were not run: 4.3 (S3) and 4.4 (subtypes) ----
    for k in ("4.3 ", "Table 9.", "4.4 ", "Table 10.", "Subtype names are", "* Random negatives"):
        drop(P[k]._p)
    drop(T[8]._tbl)
    drop(T[9]._tbl)

    # ---- 4.3 (was 4.5): FDR control ----
    heading(P["4.5 "], "4.3 Uncertainty and FDR Control")
    caption(P["Table 11."], "Table 9.", "FDR control on the S1 test set")
    s1 = t11[t11["split"] == "S1"]
    for row in T[10].rows[1:]:
        label, tgt = row.cells[0].text.strip(), row.cells[1].text.strip()
        r = s1[(s1["rule"] == "CFS") & ((s1["target_fdr"] - float(tgt)).abs() < 1e-6)] if label.startswith("CFS") \
            else s1[s1["rule"].str[:12] == label[:12]]
        fill(row.cells[2], f(r["empirical_fdr"].iloc[0]))
        fill(row.cells[3], f(r["power"].iloc[0]))
    replace_image(doc, img[12], os.path.join(FIG, "fig05_fdr_control.png"))
    caption(P["Figure 5."], "Figure 5.", "Empirical vs target FDR of CFS on the S1 test set; points on or below "
                                         "the diagonal show valid control.")
    replace_image(doc, img[13], os.path.join(FIG, "fig06_reliability_s1.png"))
    caption(P["Figure 6."], "Figure 6.", "Reliability diagram before and after temperature scaling (S1 test set).")

    # ---- 4.4 (was 4.6): explanation quality ----
    heading(P["4.6 "], "4.4 Explanation Quality")
    n_exp = int(t12["explanations"].sum())
    caption(P["Table 12."], "Table 10.", f"Explanation quality on {n_exp} AD and long COVID predictions")
    tb = T[11]
    delete_cols(tb, [col_index(tb, "Expert")])
    gen, sup = int(t12["claims_generated"].sum()), int(t12["claims_supported"].sum())
    abl, fid = csv("table12_rex_ablation.csv"), csv("table12_path_fidelity.csv")
    fd = dict(zip(fid["explainer"], fid["fidelity"]))
    ab = {r["mode"]: r for _, r in abl.iterrows()}
    rows = {r.cells[0].text.strip()[:13]: r.cells for r in tb.rows[1:]}
    c = rows["REX (Proposed"]
    fill(c[1], f(t12["citation_precision"].mean()))
    fill(c[2], "1.000")
    fill(c[3], "0.000")
    fill(c[4], f(fd["rex"]))
    c = rows["REX without v"]
    fill(c[1], f(sup / gen))
    fill(c[2], f(sup / gen))
    fill(c[3], f(1 - sup / gen))
    fill(c[4], f(fd["rex"]))
    c = rows["REX without r"]
    fill(c[2], f(ab["no_retrieval"]["claim_support_rate"]))
    fill(c[3], f(ab["no_retrieval"]["hallucination_rate"]))
    fill(c[4], f(fd["rex"]))
    c = rows["LLM-only expl"]
    fill(c[2], f(ab["llm_only"]["claim_support_rate"]))
    fill(c[3], f(ab["llm_only"]["hallucination_rate"]))
    c = rows["Path-only exp"]
    fill(c[3], "0.000")
    fill(c[4], f(fd["path_only"]))
    replace_image(doc, img[14], os.path.join(FIG, "fig07_explanations.png"))
    caption(P["Figure 7."], "Figure 7.", f"Claim support and unsupported-claim rates of REX ({gen} claims generated, "
                                         f"{sup} kept after NLI verification).")

    # ---- 4.5 (was 4.7): ablations ----
    heading(P["4.7 "], "4.5 Ablation Study")
    caption(P["Table 13."], "Table 11.", "Ablation results (S2 = neurodegenerative area held out, seed 0)")
    tb = T[12]
    delete_cols(tb, [col_index(tb, "S3")])
    delete_rows(tb, ("A4", "A5"))
    set_cell(tb.rows[0].cells[2], "S2 AUPRC (neurodeg.)", True)
    abm = {r.configuration.split(" - ")[0].split(" (")[0]: r for r in t13.itertuples(index=False)}
    for row, key in zip(tb.rows[1:], ("BRIDGE-DR", "A1", "A2", "A3", "A6")):
        fill(row.cells[1], f(abm[key].s1_auprc))
        fill(row.cells[2], f(abm[key].s2_auprc))
    replace_image(doc, img[15], os.path.join(FIG, "fig08_ablation.png"))
    caption(P["Figure 8."], "Figure 8.", "Zero-shot AUPRC (S2, neurodegenerative) of the full model and each ablation.")

    # ---- 4.6 (was 4.8): cost ----
    heading(P["4.8 "], "4.6 Computational Cost")
    caption(P["Table 14."], "Table 12.", "Cost on an RTX 3050 Laptop GPU (4 GB) with 16 GB RAM")
    tb = T[13]
    delete_rows(tb, ("TriCAL pre-training", "Docking", "TxGNN"))
    cost = {r.component: r for r in t14.itertuples(index=False)}
    cpu = {r["component"]: r for _, r in csv("table14_cost_measured_cpu.csv").iterrows()}
    gpu = {r["component"]: r for _, r in csv("table14_cost_measured_cuda.csv").iterrows()}
    find = lambda prefix: next(r for k, r in cost.items() if k.startswith(prefix))  # noqa: E731
    tr1, tr2 = find("TriCAL + BC-RGT training, S1"), find("TriCAL + BC-RGT training, S2")
    gb = lambda v: f"{float(v):.1f} GB"  # noqa: E731
    for row in tb.rows[1:]:
        label, c = row.cells[0].text.strip(), row.cells
        if label.startswith("BC-RGT"):
            set_cell(c[0], "TriCAL + BC-RGT training (per split)")
            fill(c[1], gb(tr1.peak_vram_gb))
            fill(c[2], gb(gpu["train"]["peak_ram_gb"]) if "train" in gpu else gb(cpu["train"]["peak_ram_gb"]))
            fill(c[3], f"{tr1.time} (S1), {tr2.time} (S2)")
            m = cpu["train"]["seconds_per_epoch"] / 60
            fill(c[4], f"Yes, {m:.1f} min per epoch (≈ {m * 60 / 60:.1f} h for 60 epochs)")
        elif label.startswith("Inference"):
            fill(c[1], gb(gpu["inference"]["peak_vram_gb"]) if "inference" in gpu else "< 1 GB")
            fill(c[2], gb(cpu["inference"]["peak_ram_gb"]))
            fill(c[3], f"{gpu['inference']['seconds']:.1f} s" if "inference" in gpu else find("Inference").time)
            fill(c[4], f"Yes, {cpu['inference']['seconds']:.1f} s")
        elif label.startswith("CFS"):
            fill(c[1], "0 GB (CPU)")
            fill(c[2], gb(cpu["cfs"]["peak_ram_gb"]))
            fill(c[3], f"{cpu['cfs']['seconds']:.2f} s")
            fill(c[4], "Yes (runs on CPU)")
        elif label.startswith("REX"):
            r = find("REX")
            fill(c[1], str(r.peak_vram_gb))
            fill(c[2], gb(cpu["rex"]["peak_ram_gb"]) if "rex" in cpu else "–")
            fill(c[3], r.time)
            fill(c[4], "Yes for retrieval and NLI (CPU); LLM via Ollama")
    replace_image(doc, img[16], os.path.join(FIG, "fig09_accuracy_vs_cost.png"))
    caption(P["Figure 9."], "Figure 9.", "Zero-shot AUPRC vs training time (bubble size = peak VRAM) for the full "
                                         "model and the ablations.")
    replace_image(doc, img[17], os.path.join(FIG, "fig10_overall_comparison.png"))
    caption(P["Figure 10."], "Figure 10.", "Overall comparison of the full model and the ablations across seven "
                                           "criteria (min–max normalised).")

    # ---- 4.7 (was 4.9): candidates ----
    heading(P["4.9 "], "4.7 Candidate Drugs")
    caption(P["Table 15."], "Table 13.", "Top-10 candidates for Alzheimer disease (S1 model)")
    chk = {(r.disease, r.drug): r for r in csv("table15_16_candidate_checks.csv").itertuples(index=False)}
    imm = {(r.disease, r.drug): r.label for r in csv("table16_immune_targets.csv").itertuples(index=False)}

    def fill_candidates(table, df, col7):
        resize_rows(table, len(df))
        for row, r in zip(table.rows[1:], df.itertuples(index=False)):
            c, k = row.cells, (r.disease, r.drug)
            set_cell(c[0], int(r.rank))
            path = r.key_path.replace(" -> ", " → ") if isinstance(r.key_path, str) and r.key_path else "no KG path found"
            pmids = r.pmids if isinstance(r.pmids, str) and r.pmids else "none retrieved"
            for j, v in ((1, r.drug), (3, f"{r.score:.3f}"), (4, f"{r.conformal_p:.4f}"), (5, path), (6, pmids),
                         (7, col7(k)), (9, chk[k].trial_status)):
                fill(c[j], v)
        delete_cols(table, [2, 8])  # subtype and docking columns were not run

    def bbb(k):
        c = chk[k]
        return "no structure (biologic)" if not c.has_structure else ("yes" if c.bbb_rule else "no") + f" (TPSA {c.tpsa:.0f})"

    t16_el = copy.deepcopy(T[14]._tbl)
    fill_candidates(T[14], t15, bbb)
    caption(P["Table 16."], "Table 14.", "Top-10 candidates for long COVID-19 (S1 model)")
    P["Table 16."]._p.addnext(t16_el)
    t16_tbl = docx.table.Table(t16_el, P["Table 16."]._parent)
    set_cell(t16_tbl.rows[0].cells[7], "Immune / endothelial target", True)
    fill_candidates(t16_tbl, t16, lambda k: imm[k])
    set_par(P["Same columns as Table 15"], (
        "Tables 13–14: BBB+ is a rule on the PubChem structure (TPSA < 90 Å², MW < 450, HBD ≤ 3); immune / endothelial "
        "target = a PrimeKG drug target linked to an immune, inflammatory or endothelial GO process; trial status = "
        "ClinicalTrials.gov studies listing the drug as an intervention for the condition (long COVID: Post-Acute "
        "COVID-19 Syndrome). No candidate passed the per-disease CFS shortlist at FDR 0.10, so the lists are ranked "
        "hypotheses.", False))
    plain_paragraph(P["Same columns as Table 15"])
    t16_el.addnext(P["Same columns as Table 15"]._p)

    # ---- 4.8 (was 4.10): figures from the implementation ----
    heading(P["4.10 "], "4.8 Additional Figures")
    for p in [p for p in doc.paragraphs if re.match(r"Figure 1[1-6]:", p.text.strip())]:
        drop(p._p)
    anchor = P["4.10 "]._p
    for n, png, text in (
            (11, "fig11_drug_embedding_tsne.png", "t-SNE of learned drug embeddings, coloured by each drug's dominant "
                                                  "indication area."),
            (12, "fig14_rex_subgraph.png", "REX path subgraph for the first Alzheimer disease candidate (top KG paths by "
                                           "attention × gradient)."),
            (13, "fig16_funnel.png", "Candidate funnel: all drugs → CFS shortlist → top-10 → drug-likeness, BBB and "
                                     "trial checks → REX → verified claims.")):
        fp = insert_figure_after(doc, anchor, fig_tmpl, os.path.join(FIG, png))
        cp = insert_after(fp._p, cap_tmpl, (f"Figure {n}.", True), (" " + text, False))
        mark(cp, TITLE_MARK)
        anchor = cp._p

    # ---- green titles for the implemented parts of sections 2-3 ----
    for p in doc.paragraphs:
        if p.text.strip().startswith(("2.1 Primary Dataset", "2.3 Data Splits", "Stage 1:", "Stage 2:", "Stage 3:",
                                      "Stage 4:", "Stage 5:", "Baselines, Ablations and Metrics")):
            mark(p, TITLE_MARK)

    ctx = dict(doc=doc, head1=head1, head2=head2, tab_cap=tab_cap_tmpl, fig_cap=cap_tmpl, body=body_tmpl)
    if WITH_LITERATURE:
        literature(**ctx)
    appendix(**ctx)
    data_availability(**ctx)

    for i, dp in enumerate(doc.element.body.iter(qn("wp:docPr")), start=1):  # unique drawing ids
        dp.set("id", str(i))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        doc.save(OUT)
        print(f"Saved {OUT}")
    except PermissionError:  # the report is open in Word
        alt = OUT.replace(".docx", "_updated.docx")
        doc.save(alt)
        print(f"{OUT} is open in another program; saved {alt} instead")


class Writer:
    """Appends headings, paragraphs, tables and figures at the end of the document."""

    def __init__(self, doc, head1, head2, tab_cap, fig_cap, body, prefix=""):
        self.doc, self.h1t, self.h2t, self.tct, self.fct, self.bt = doc, head1, head2, tab_cap, fig_cap, body
        self.prefix, self.nt, self.nf = prefix, 0, 0

    def h1(self, text, new_page=True):
        p = add_copy(self.doc, self.h1t, (text, False))
        mark(p, TITLE_MARK)
        p.paragraph_format.page_break_before = new_page

    def h2(self, text):
        mark(add_copy(self.doc, self.h2t, (text, False)), TITLE_MARK)

    def para(self, text):
        add_copy(self.doc, self.bt, (text, False))

    def table(self, title, header, rows, widths=None):
        self.nt += 1
        mark(add_copy(self.doc, self.tct, (f"Table {self.prefix}{self.nt}.", True), (" " + title, False)), TITLE_MARK)
        add_table(self.doc, header, rows, widths)

    def figure(self, png, title, width_in=6.0):
        if add_figure(self.doc, os.path.join(FIG, png), width_in) is None:
            return
        self.nf += 1
        mark(add_copy(self.doc, self.fct, (f"Figure {self.prefix}{self.nf}.", True), (" " + title, False)), TITLE_MARK)


def literature(doc, head1, head2, tab_cap, fig_cap, body):
    w = Writer(doc, head1, head2, tab_cap, fig_cap, body)
    w.nt, w.nf = 14, 13  # continue the numbering of section 4
    w.h1("5. LITERATURE REVIEW AND COMPARISON")
    w.h2("5.1 Drug repurposing and biomedical knowledge graphs")
    w.para("Drug repurposing finds new uses for approved or investigational drugs and can shorten development "
           "because safety data already exist [1]. Knowledge graphs make this a link-prediction problem. Hetionet "
           "integrated 29 public resources into 47,031 nodes and 2,250,197 relationships and ranked 209,168 "
           "compound–disease pairs [3]. PrimeKG, used here, describes 17,080 diseases with 4,050,249 relationships "
           "and is rich in indication, contraindication and off-label drug–disease edges [2].")
    w.h2("5.2 Knowledge-graph embedding and graph neural networks")
    w.para("Embedding models score a triple from learned entity and relation vectors: TransE as a translation [5], "
           "DistMult as a bilinear product [6] and RotatE as a rotation in complex space [7]. Message-passing models "
           "learn node representations from the graph: R-GCN with relation-specific weights [8], CompGCN with "
           "composition operators [9] and HGT with type-dependent attention [10]. NBFNet generalises path-based "
           "methods for link prediction [11]. All of these except NBFNet were reimplemented in the same framework and "
           "evaluated on the same splits (Tables 7–8, Appendix A).")
    w.h2("5.3 Zero-shot drug repurposing")
    w.para("TxGNN ranks drugs as indications and contraindications for 17,080 diseases with a graph neural network and "
           "a metric-learning module, and reports improvements of 49.2% for indications and 35.1% for "
           "contraindications over 8 methods under zero-shot evaluation, together with multi-hop path explanations "
           "[4]. Recent PrimeKG-based and multimodal models extend this line: MRDGNN reasons over relational digraphs "
           "and uses contraindications as complementary evidence [19], and HAMGNN combines heterogeneous attention, "
           "disease-level meta-learning and LLM-extracted literature, reporting ROC-AUC 0.98 under a disjoint-disease "
           "protocol with Alzheimer's disease and long COVID case studies [18].")
    w.h2("5.4 Transcriptomic signature reversal")
    w.para("The Connectivity Map links drugs and diseases through gene-expression signatures [12], and its L1000 "
           "extension contains about 1.3 million profiles [13]. BRIDGE-DR's signature-reversal loss (SRCL) is built "
           "for these data; L1000 profiles were not supplied for this run, so SRCL was inactive.")
    w.h2("5.5 Uncertainty, calibration and false-discovery control")
    w.para("Neural scores are often miscalibrated, and temperature scaling is a simple remedy [16]. Conformal "
           "selection turns scores into p-values against known nulls and selects candidates with a "
           "Benjamini–Hochberg procedure [14, 15], which bounds the false-discovery rate of the shortlist. BRIDGE-DR "
           "applies this to drug shortlists (CFS); the measured FDR stayed at or below each target on S1 (Table 9).")
    w.h2("5.6 Grounded explanations")
    w.para("Retrieval-augmented generation conditions a language model on retrieved documents [17]. TxGNN explains "
           "predictions with multi-hop knowledge paths [4]. REX combines both and adds automatic NLI verification of "
           "every claim against its cited abstract; without retrieval only 2% of the claims of a 3B model were "
           "supported (Table 10).")
    w.h2("5.7 Positioning of this work")
    w.table("Capabilities of BRIDGE-DR and related drug-repurposing systems (as described in each paper)",
            ["System", "Knowledge source", "Unseen diseases", "Uncertainty / FDR control", "Explanations"], [
                ["Rephetio / Hetionet [3]", "Hetionet (29 resources)", "Not the focus", "Not reported", "Network patterns"],
                ["Connectivity Map [12, 13]", "Expression signatures", "Signature-based", "Not reported",
                 "Signature connectivity"],
                ["TxGNN [4]", "PrimeKG", "Zero-shot (17,080 diseases)", "Not reported", "Multi-hop paths, human evaluation"],
                ["MRDGNN [19]", "PrimeKG", "Held-out drugs (cold start)", "Not reported", "Relational evidence"],
                ["HAMGNN [18]", "DrugBank, DisGeNET, Hetionet, literature", "Disjoint diseases (cold start)",
                 "Not reported", "Mechanistic pathways"],
                ["BRIDGE-DR (this work)", "PrimeKG + drug structures", "Zero-shot (9 areas, Table 8)",
                 "Conformal p-values + BH (Table 9)", "KG paths + PubMed, NLI-verified (Table 10)"],
            ], widths=[1.4, 1.5, 1.3, 1.3, 1.5])
    methods = csv("appendix_methods.csv")
    br = methods[methods["method"] == "BRIDGE-DR"].set_index("split")
    sc = csv("seed_check_s2.csv")
    bz = sc[sc["method"] == "BRIDGE-DR"]["auprc"]
    w.table("Reported results of related work and the results of this implementation",
            ["Study", "Data", "Protocol", "Result"], [
                ["TxGNN [4]", "PrimeKG", "Zero-shot disease split", "+49.2% indications, +35.1% contraindications "
                                                                     "vs 8 methods (relative)"],
                ["HAMGNN [18]", "DrugBank, DisGeNET, Hetionet (> 2.2M edges)", "Disjoint diseases",
                 "ROC-AUC 0.98, precision 0.95"],
                ["BRIDGE-DR (this work)", "PrimeKG (2.26M edges after capping)", "S1 random 80/10/10",
                 f"AUROC {br.loc['S1', 'auroc']:.3f}, AUPRC {br.loc['S1', 'auprc']:.3f}"],
                ["BRIDGE-DR (this work)", "PrimeKG", "S2 neurodegenerative held out",
                 f"AUROC {br.loc['S2', 'auroc']:.3f}, AUPRC {br.loc['S2', 'auprc']:.3f} (seed 0); AUPRC "
                 f"{bz.mean():.3f} ± {bz.std():.3f} over 2 seeds"],
            ], widths=[1.4, 1.8, 1.5, 2.3])
    w.para("Reported numbers come from different graphs, splits and negative sampling and are not directly comparable "
           "with each other or with this run; the comparison on equal terms is the set of reimplemented baselines in "
           "Tables 7–8 and Appendix A.")
    w.h1("REFERENCES", new_page=False)
    for i, ref in enumerate(REFERENCES, start=1):
        w.para(f"[{i}] {ref}")


def appendix(doc, head1, head2, tab_cap, fig_cap, body):
    w = Writer(doc, head1, head2, tab_cap, fig_cap, body, prefix="A")
    if WITH_LITERATURE:
        w.h1("APPENDIX A. ADDITIONAL RESULTS FROM THE IMPLEMENTATION")
        w.para("Everything in this appendix was produced by the implementation (PrimeKG, laptop RTX 3050 4 GB, seed 0 "
               "unless stated). Tables and figures are drawn from the CSVs in reports/figure_data/ and can be reused "
               "in a manuscript.")

    w.h2("A.1 Implementation settings")
    w.table("Settings used in this implementation run", ["Item", "As implemented"], [
        ["Code", "bridge_dr/ package (data, TriCAL, BC-RGT, SRCL, CFS, REX, metrics); scripts/run.py runs the full loop"],
        ["Hardware", "NVIDIA RTX 3050 Laptop 4 GB, 16 GB RAM, Windows 11; Python 3.11, PyTorch 2.7 (CUDA 11.8)"],
        ["Graph", "PrimeKG; each structural relation capped at 150,000 edges (2.26M edges, 44 directed relation types "
                  "+ 4 label relations); 7,957 drugs, 17,080 diseases, 11,831 indications, 30,675 contraindications"],
        ["Splits", "S1 random 80/10/10 of indications; S2 all indications of one disease area held out (9 areas)"],
        ["Drug features", "Morgan fingerprints (radius 2, 2,048 bits) from PubChem SMILES for 1,846 drugs"],
        ["Disease signatures", "KG gene neighbourhood over the 512 most disease-associated genes"],
        ["TriCAL", "2-layer MLP projections, symmetric InfoNCE (τ = 0.07), modality dropout 0.3, gated fusion"],
        ["BC-RGT", "2 relational graph-transformer layers, hidden 128, 4 heads; 300k edges sampled per step; FiLM "
                   "conditioning; focal loss + 0.2·TriCAL"],
        ["Training", "AdamW (lr 1e-3, cosine), ≤ 60 epochs, early stopping on validation AUPRC (patience 8)"],
        ["CFS", "Temperature scaling; conformal p-values against held-out contraindications; Benjamini–Hochberg"],
        ["REX", "KG paths (≤ 3 hops, attention × gradient); PubMed + BM25; qwen2.5:3b via Ollama; DeBERTa-v3 NLI"],
        ["Baselines", "R-GCN, CompGCN, HGT, TransE, DistMult, RotatE on the same splits; LLM-only ranking (qwen2.5:3b)"],
        ["Candidate checks", "Lipinski and BBB rules (RDKit); PrimeKG immune / endothelial targets; ClinicalTrials.gov"],
    ], widths=[1.4, 5.0])

    m = csv("appendix_methods.csv")
    kg = m[m["method"] != "LLM-only (qwen2.5:3b)"]
    w.h2("A.2 All implemented methods on S1 and S2")
    w.table("BRIDGE-DR and the reimplemented baselines (seed 0)",
            ["Method", "Split", "AUROC", "AUPRC", "Hits@10", "MRR", "Training (min)"],
            [[r["method"], r["split"], f(r["auroc"]), f(r["auprc"]), f(r["hits@10"]), f(r["mrr"]), f(r["minutes"], 1)]
             for _, r in kg.iterrows()], widths=[1.6, 0.5, 0.7, 0.7, 0.7, 0.7, 0.9])
    llm = m[m["method"] == "LLM-only (qwen2.5:3b)"]
    if len(llm):
        w.table("LLM-only ranking without the knowledge graph (qwen2.5:3b, 450 sampled test pairs per split)",
                ["Split", "AUROC", "AUPRC", "Inference (min)"],
                [[r["split"], f(r["auroc"]), f(r["auprc"]), f(r["minutes"], 1)] for _, r in llm.iterrows()],
                widths=[0.8, 1, 1, 1.2])
    w.figure("figA1_methods_s1_s2.png", "AUPRC of every implemented method on S1 and S2 (neurodegenerative).")
    w.figure("figA4_roc_pr_all_methods.png", "ROC and precision–recall curves of the implemented methods on S1 and "
                                             "S2, rebuilt from the saved test scores (random negatives redrawn, so "
                                             "areas differ from Table A2 by at most 0.02).", 6.3)
    w.figure("figA5_training_curves_all_methods.png", "Validation AUPRC per epoch for the implemented methods.", 6.3)
    w.figure("figA6_accuracy_vs_cost_all_methods.png", "Test AUPRC vs training time of the implemented methods.", 6.3)

    a, rg = csv("appendix_areas.csv"), csv("rgcn_zero_shot_areas.csv")
    w.h2("A.3 Zero-shot results for every held-out disease area")
    rgd = {r["area"]: r for _, r in rg.iterrows()}
    both = all(x in rgd for x in a["area"])
    hdr = ["Held-out area", "AUROC", "AUPRC", "MRR", "Hits@100", "Median rank"] + (["R-GCN AUROC", "R-GCN AUPRC"] if both else [])
    w.table("BRIDGE-DR (and R-GCN) on S2 with each disease area held out (seed 0)", hdr,
            [[r["area"], f(r["auroc"]), f(r["auprc"]), f(r["mrr"]), f(r["hits@100"]), f"{r['median_rank']:.0f}"]
             + ([f(rgd[r["area"]]["auroc"]), f(rgd[r["area"]]["auprc"])] if both else []) for _, r in a.iterrows()],
            widths=[1.5, 0.6, 0.6, 0.6, 0.7, 0.8] + ([0.8, 0.8] if both else []))
    w.figure("figA3_zero_shot_areas.png", "Zero-shot AUROC and AUPRC of BRIDGE-DR per held-out area; the dashed line "
                                          "marks chance-level AUROC.")
    if both:
        w.figure("figA9_zero_shot_bridge_vs_rgcn.png", "Zero-shot AUPRC per held-out area: BRIDGE-DR vs R-GCN.")

    sc = csv("seed_check_s2.csv")
    w.h2("A.4 Seed check on the zero-shot split")
    w.table("S2 (neurodegenerative) over two seeds", ["Method", "AUPRC seed 0", "AUPRC seed 1", "AUPRC mean ± SD",
                                                     "AUROC mean ± SD"],
            [[meth, *(f(v) for v in g["auprc"]), f"{g['auprc'].mean():.3f} ± {g['auprc'].std():.3f}",
              f"{g['auroc'].mean():.3f} ± {g['auroc'].std():.3f}"] for meth, g in sc.groupby("method", sort=False)],
            widths=[1.4, 1.1, 1.1, 1.4, 1.4])
    w.figure("figA2_seed_check.png", "S2 AUPRC per seed; diamonds show mean ± SD.", 4.2)

    abl_rows = []
    for d, name in (("primekg", "BRIDGE-DR (full)"), ("ablations/A1_no_tricals", "A1 – no TriCAL"),
                    ("ablations/A2_no_modality_dropout", "A2 – no modality dropout"),
                    ("ablations/A3_no_subtype_conditioning", "A3 – no subtype conditioning"),
                    ("ablations/A6_random_negatives_only", "A6 – random negatives only")):
        with open(os.path.join(ROOT, "results", d, "results.json"), encoding="utf-8") as fh:
            rs = {r["split"]: r for r in json.load(fh)}
        abl_rows.append([name, f(rs["s1"]["test"]["auroc"]), f(rs["s1"]["test"]["auprc"]), f(rs["s2"]["test"]["auroc"]),
                         f(rs["s2"]["test"]["auprc"]), f(rs["s2"]["fit"]["train_seconds"] / 60, 1)])
    w.h2("A.5 Ablations with AUROC and AUPRC")
    w.table("Ablations on S1 and S2 (neurodegenerative), seed 0", ["Configuration", "S1 AUROC", "S1 AUPRC", "S2 AUROC",
                                                                 "S2 AUPRC", "S2 training (min)"], abl_rows,
            widths=[2.0, 0.8, 0.8, 0.8, 0.8, 1.0])

    cal = csv("appendix_calibration.csv")
    w.h2("A.6 Calibration and score distributions")
    w.table("Temperature scaling fitted on validation data", ["Split", "Temperature", "ECE before", "ECE after",
                                                             "Brier before", "Brier after"],
            [[r["split"], f(r["temperature"]), f(r["ece_before"]), f(r["ece_after"]), f(r["brier_before"]),
              f(r["brier_after"])] for _, r in cal.iterrows()], widths=[0.7, 1, 1, 1, 1, 1])
    w.figure("figA7_score_distribution.png", "BRIDGE-DR scores of test indications, random negatives and held-out "
                                             "contraindications.", 6.3)

    w.h2("A.7 Explanation quality in detail")
    t12, rx, fid = csv("table12_fig07_explanations.csv"), csv("table12_rex_ablation.csv"), csv("table12_path_fidelity.csv")
    rows = [[r["model"], int(r["explanations"]), int(r["claims_generated"]), int(r["claims_supported"]),
             f(r["claim_support_rate"])] for _, r in t12.iterrows()]
    rows += [[{"no_retrieval": "REX without retrieval", "llm_only": "LLM-only explanation"}[r["mode"]],
              int(r["explanations"]), int(r["claims"]), int(r["supported"]), f(r["claim_support_rate"])]
             for _, r in rx.iterrows()]
    w.table("Claims and NLI support (qwen2.5:3b, DeBERTa-v3 NLI)", ["Setting", "Explanations", "Claims",
                                                                   "NLI-supported", "Support rate"], rows,
            widths=[2.0, 1, 0.8, 1.1, 1])
    names = {"rex": "REX paths (attention × gradient)", "path_only": "Path-only (graph order)",
             "random": "Random edges at the drug (control)"}
    w.table(f"Path fidelity: relative score drop when the explanation's edges are removed "
            f"({int(fid['n_predictions'].iloc[0])} predictions with paths)", ["Edges removed", "Path fidelity"],
            [[names[r["explainer"]], f(r["fidelity"])] for _, r in fid.iterrows()], widths=[3.0, 1.2])
    w.figure("figA8_path_fidelity.png", "Path fidelity of REX paths, model-agnostic paths and random edges.", 4.5)

    chk = csv("table15_16_candidate_checks.csv")
    w.h2("A.8 Drug-likeness, BBB and trial status of the candidates")
    yn = lambda v: "yes" if str(v) == "True" else "no"  # noqa: E731
    w.table("Checks on the top-10 candidates (S1 model); biologics have no small-molecule structure",
            ["Disease", "Rank", "Drug", "MW", "TPSA", "Lipinski", "Drug-like", "BBB+", "Trials"],
            [[r["disease"], int(r["rank"]), r["drug"]] + ([f"{r['mw']:.0f}", f"{r['tpsa']:.0f}",
                                                           int(r["lipinski_violations"]), yn(r["drug_like"]),
                                                           yn(r["bbb_rule"])] if r["has_structure"]
                                                          else ["biologic"] * 5) + [r["trial_status"]]
             for _, r in chk.iterrows()], widths=[1.1, 0.4, 1.3, 0.5, 0.5, 0.6, 0.6, 0.5, 1.3])

    w.h2("A.9 Training curves")
    w.figure("figS1_training_curves.png", "Training loss and validation AUPRC per epoch of BRIDGE-DR for S1 and S2.")

    if WITH_LITERATURE:
        w.h2("A.10 Scope of this run")
        for text in (
            "Single seed (seed 0, plus seed 1 for the S2 check) with a reduced model (hidden 128, 2 layers) on a 4 GB "
            "laptop GPU.",
            "Not run because they need more GPU memory or data that was not supplied: TxGNN and NBFNet (16–40 GB GPU), "
            "CMap reversal and SRCL (L1000 profiles), the time-sliced test S3 (AACT trial dates), subtype analysis (GEO "
            "cohorts), docking, expert scoring and an 8B LLM. Their rows, columns and sections are not reported.",
            "Table 7 lists the methods run in this implementation; the LLM-only ranking (pairwise scores only) is in "
            "Table A3.",
            "Table 10: after verification every kept claim is NLI-supported by construction; the raw rate is in the "
            "'without verification' row. Path fidelity uses the same REX paths for the first three rows.",
        ):
            w.para("• " + text)


def data_availability(doc, head1, head2, tab_cap, fig_cap, body):
    """Original sources of every dataset, model and service used, for a manuscript's availability statement."""
    w = Writer(doc, head1, head2, tab_cap, fig_cap, body)
    w.nt = 16 if WITH_LITERATURE else 14  # continue the numbering of the main text
    w.h1("6. DATA AND CODE AVAILABILITY" if WITH_LITERATURE else "5. DATA AND CODE AVAILABILITY")
    w.para("All data used in this study are publicly available from their original sources below. PrimeKG (kg.csv) was "
           "downloaded from Harvard Dataverse on 28 October 2026 and used unchanged apart from the pruning described in "
           "Table A1.")
    w.table("Original sources of the data, models and services used", ["Resource", "Used for", "Original link"], [
        ["PrimeKG knowledge graph, Harvard Dataverse (Chandak et al., Sci Data 2023)",
         "Graph, indications, contraindications", "https://doi.org/10.7910/DVN/IXA7BM"],
        ["PrimeKG kg.csv (Dataverse file 6180620)", "Direct download of the edge list",
         "https://dataverse.harvard.edu/api/access/datafile/6180620"],
        ["PrimeKG code and documentation", "Construction and description of the graph",
         "https://github.com/mims-harvard/PrimeKG"],
        ["PubChem PUG-REST", "Drug structures (SMILES) for Morgan fingerprints and BBB / Lipinski checks",
         "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest"],
        ["NCBI E-utilities (PubMed)", "Abstract retrieval for REX", "https://www.ncbi.nlm.nih.gov/books/NBK25501/"],
        ["ClinicalTrials.gov API v2", "Trial status of the candidates", "https://clinicaltrials.gov/data-api/api"],
        ["Qwen2.5 3B via Ollama", "LLM for REX and the LLM-only baseline", "https://ollama.com/library/qwen2.5"],
        ["cross-encoder/nli-deberta-v3-small", "NLI verification of REX claims",
         "https://huggingface.co/cross-encoder/nli-deberta-v3-small"],
    ], widths=[1.8, 1.7, 3.4])
    w.para("Code, trained models (checkpoints/) and the data behind every table and figure (reports/figure_data/) are in "
           "the BRIDGE-DR repository, https://github.com/jeevatechoffice500k/bridge-dr (private; access on request), "
           "which includes step-by-step instructions to reproduce the results.")


if __name__ == "__main__":
    main()
