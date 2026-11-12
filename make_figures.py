"""Draw report figures from the editable CSVs in reports/figure_data/ -> reports/figures/*.png.

Edit any CSV (e.g. to correct a value or add a baseline row) and re-run this script.
A figure whose CSV is missing or empty is skipped.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reports", "figure_data")
FIG = os.path.join(ROOT, "reports", "figures")
PALETTE = ["#2b7bd6", "#e8643c", "#1fae7f", "#f0a30a", "#e57fa6", "#7f63c9", "#8a8a8a"]
BLUES = LinearSegmentedColormap.from_list("b", ["#d6e6fa", "#3d88e0", "#173f7f"])

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 11, "axes.edgecolor": "#666666",
    "axes.labelcolor": "#222222", "xtick.color": "#666666", "ytick.color": "#666666",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5",
    "legend.frameon": False,
})


def csv(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def measured(src) -> bool:
    return str(src).strip().lower().startswith("measured")


def fig02():
    df, tab, base = csv("fig02_pr_curve_s1.csv"), csv("table07_s1_prediction.csv"), csv("baselines_s1.csv")
    if df is None:
        return
    auprc = dict(zip(tab["method"], tab["auprc"])) if tab is not None else {}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.05, 1]})
    for i, (m, g) in enumerate(df.groupby("method", sort=False)):
        ax.plot(g["recall"], g["precision"], color=PALETTE[i % 7], lw=2.6 if i == 0 else 1.8,
                label=f"{m} (AUPRC {auprc.get(m, float('nan')):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("(a) Measured precision–recall curves", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    # (b) AUPRC: measured BRIDGE-DR vs comparison methods (hatched = plan target, not measured)
    rows = [("BRIDGE-DR (measured)", tab["auprc"].iloc[0], True)] if tab is not None else []
    if base is not None:
        rows += [(m, v, measured(s)) for m, v, s in zip(base["method"], base["auprc"], base["source"])]
    rows = sorted(rows, key=lambda r: r[1])
    y = np.arange(len(rows))
    for yi, (m, v, meas) in zip(y, rows):
        is_ours = m.startswith("BRIDGE-DR")
        bx.barh(yi, v, color=PALETTE[0] if is_ours else ("#9a9a9a" if meas else "white"),
                edgecolor=PALETTE[0] if is_ours else "#7a7a7a", hatch=None if meas else "///", lw=1)
        bx.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    bx.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    bx.set_xlim(min(0.5, np.floor(min(r[1] for r in rows) * 10) / 10 - 0.05), 1.0)
    bx.set_xlabel("AUPRC (S1)")
    bx.grid(axis="y", visible=False)
    bx.set_title("(b) AUPRC; hatched = value from the plan, not measured", fontsize=10)
    save(fig, "fig02_pr_curve_s1.png")


def fig03():
    """Zero-shot AUPRC per held-out area for the methods run on every area (BRIDGE-DR, R-GCN)."""
    a, r = csv("appendix_areas.csv"), csv("rgcn_zero_shot_areas.csv")
    if a is None:
        return
    areas = list(a["area"])
    labels, mat = ["BRIDGE-DR"], [list(a["auprc"])]
    if r is not None and set(areas) <= set(r["area"]):
        labels.append("R-GCN")
        mat.append([dict(zip(r["area"], r["auprc"]))[x] for x in areas])
    mat = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 1.4 + 0.6 * len(labels)))
    im = ax.imshow(mat, cmap=BLUES, vmin=0.3, vmax=1.0, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(areas)), areas, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#222222" if v <= 0.62 else "white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("AUPRC")
    ax.set_title("Zero-shot AUPRC per held-out disease area (S2, seed 0)", fontsize=10)
    save(fig, "fig03_zero_shot_heatmap.png")


def fig04():
    df = csv("fig04_recall_at_k.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    for i, (s, g) in enumerate(df.groupby("split", sort=False)):
        ax.plot(g["k"], g["recall"], "o-", color=PALETTE[i], lw=2.2, label=s)
    ax.set_xlabel("k (top-k drugs per disease)")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig04_recall_at_k.png")


def fig05():
    df = csv("table11_fig05_fdr.csv")
    if df is None:
        return
    df = df[(df["rule"] == "CFS") & (df["shortlist"] > 0)]
    fig, ax = plt.subplots(figsize=(4.9, 4.1))
    ax.plot([0, 0.25], [0, 0.25], "--", color="#999999", lw=1.2, label="Target = empirical")
    for i, (s, g) in enumerate(df.groupby("split")):
        ax.plot(g["target_fdr"], g["empirical_fdr"], "o-", color=PALETTE[i], ms=8, lw=2, label=f"CFS ({s})")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("Target FDR (α)")
    ax.set_ylabel("Empirical FDR")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig05_fdr_control.png")


def fig06():
    df = csv("fig06_reliability_s1.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(4.9, 4.1))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.2, label="Perfect calibration")
    for i, (s, g) in enumerate(df.groupby("scaling", sort=False)):
        ax.plot(g["mean_predicted"], g["fraction_positive"], "o-", color=PALETTE[i], lw=2,
                label="Before temperature scaling" if s == "before" else "After temperature scaling")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig06_reliability_s1.png")


def fig07():
    df = csv("table12_fig07_explanations.csv")
    if df is None:
        return
    metrics = [("claim_support_rate", "Claim support rate"),
               ("hallucination_rate_before_verification", "Unsupported claims\nbefore verification")]
    x = np.arange(len(metrics))
    w = 0.8 / len(df)
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    for i, row in enumerate(df.itertuples(index=False)):
        vals = [getattr(row, m) for m, _ in metrics]
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, w, color=PALETTE[i], label=row.model)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.legend(fontsize=9)
    save(fig, "fig07_explanations.png")


def fig08():
    df = csv("table13_fig08_09_10_ablations.csv")
    if df is None:
        return
    df = df.dropna(subset=["s2_auprc"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = [PALETTE[0] if c.startswith("BRIDGE") else "#9db9e0" for c in df["configuration"]]
    b = ax.barh(df["configuration"][::-1], df["s2_auprc"][::-1], color=colors[::-1])
    ax.bar_label(b, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xlabel("Zero-shot AUPRC (S2, neurodegenerative)")
    ax.set_xlim(0, max(df["s2_auprc"]) * 1.15)
    ax.grid(axis="y", visible=False)
    save(fig, "fig08_ablation.png")


def fig09():
    df = csv("table13_fig08_09_10_ablations.csv")
    if df is None:
        return
    df = df.dropna(subset=["s2_auprc", "train_min_s2"])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, row in enumerate(df.itertuples(index=False)):
        ax.scatter(row.train_min_s2, row.s2_auprc, s=300 * row.peak_vram_gb, color=PALETTE[i % 7], alpha=0.85,
                   edgecolor="white", lw=1.5)
        ax.annotate(f"{row.configuration}\n{row.peak_vram_gb:.1f} GB VRAM", (row.train_min_s2, row.s2_auprc),
                    xytext=(14, 0), textcoords="offset points", fontsize=8, va="center")
    ax.set_xlabel("Training time, S2 run on RTX 3050 4 GB (minutes)")
    ax.set_ylabel("Zero-shot AUPRC (S2)")
    ax.set_xlim(0, max(df["train_min_s2"]) * 1.6)
    save(fig, "fig09_accuracy_vs_cost.png")


def fig10():
    df = csv("table13_fig08_09_10_ablations.csv")
    if df is None:
        return
    crit = [("s1_auroc", "S1 AUROC"), ("s1_auprc", "S1 AUPRC"), ("s2_auroc", "S2 AUROC"), ("s2_auprc", "S2 AUPRC"),
            ("s1_hits@100", "S1 Hits@100"), ("s1_cfs_power@0.10", "CFS power\n(FDR 0.10)"), ("train_min_s1", "Speed")]
    df = df.dropna(subset=[c for c, _ in crit])
    if not len(df):
        return
    vals = df[[c for c, _ in crit]].astype(float).copy()
    vals["train_min_s1"] = 1 / vals["train_min_s1"]
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12) * 0.8 + 0.2
    ang = np.linspace(0, 2 * np.pi, len(crit), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(5.2, 5.3), subplot_kw={"polar": True})
    for i, (row, cfg) in enumerate(zip(norm.values, df["configuration"])):
        v = row.tolist() + row.tolist()[:1]
        ax.plot(ang + ang[:1], v, color=PALETTE[i % 7], lw=2.4 if i == 0 else 1.4, label=cfg)
        if i == 0:
            ax.fill(ang + ang[:1], v, color=PALETTE[0], alpha=0.12)
    ax.set_xticks(ang, [c[1] for c in crit], fontsize=9)
    ax.set_yticklabels([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8)
    save(fig, "fig10_overall_comparison.png")


def fig11():
    df = csv("fig11_drug_embedding_tsne.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    order = df["dominant_area"].value_counts().index
    for i, a in enumerate(order):
        g = df[df["dominant_area"] == a]
        ax.scatter(g["x"], g["y"], s=7, alpha=0.7, color="#c8c8c8" if a == "Other" else plt.cm.tab10(i % 10),
                   label=f"{a} ({len(g)})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7.5, markerscale=2.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig11_drug_embedding_tsne.png")


def fig14():
    df = csv("fig14_rex_subgraph.csv")
    if df is None:
        return
    import networkx as nx

    G = nx.DiGraph()
    for r in df.itertuples(index=False):
        G.add_edge(r.source, r.target, relation=r.relation.split(":")[0])
    drug, disease = df["drug"].iloc[0], df["target"].iloc[-1]
    depth = {drug: 0, disease: 3}
    for r in df.itertuples(index=False):
        hops = df[df["path"] == r.path]
        for k, n in enumerate(hops["source"]):
            depth.setdefault(n, k)
    layers: dict[int, list] = {}
    for n in G.nodes:
        layers.setdefault(depth.get(n, 2), []).append(n)
    pos = {n: (3.2 * d, (len(ns) - 1) / 2 - i) for d, ns in layers.items() for i, n in enumerate(ns)}
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    col = {n: PALETTE[0] if n == drug else PALETTE[1] if n == disease else PALETTE[2] for n in G.nodes}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#aaaaaa", arrows=True, arrowsize=12, node_size=2600,
                           min_source_margin=28, min_target_margin=28)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=nx.get_edge_attributes(G, "relation"), font_size=6.5,
                                 label_pos=0.5, bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"})
    for n, (x, y) in pos.items():
        ax.text(x, y, n, ha="center", va="center", fontsize=8.5, color="white",
                bbox={"boxstyle": "round,pad=0.45", "fc": col[n], "ec": "none"})
    ax.set_title(f"REX paths: {drug} → {disease}", fontsize=11)
    ax.axis("off")
    save(fig, "fig14_rex_subgraph.png")


def fig16():
    df = csv("fig16_funnel.csv")
    if df is None:
        return
    groups = list(df.groupby("disease", sort=False))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 0.4 * df.groupby("disease").size().max() + 0.8))
    axes = np.atleast_1d(axes)
    for ax, (d, g) in zip(axes, groups):
        vals = g["count"].values
        y = np.arange(len(g))[::-1]
        ax.barh(y, np.maximum(vals, 0.8), color=PALETTE[0], alpha=0.85)
        ax.set_xscale("log")
        ax.set_yticks(y, g["stage"], fontsize=8.5)
        for yi, v in zip(y, vals):
            ax.text(max(v, 0.8) * 1.15, yi, f"{v:,}", va="center", fontsize=8.5)
        ax.set_xlim(0.7, vals.max() * 6)
        ax.set_xlabel("Number of drugs (log scale)")
        ax.grid(axis="y", visible=False)
        ax.set_title(d, fontsize=10)
    save(fig, "fig16_funnel.png")


def figA1():
    """Measured methods: S1 vs S2 AUPRC side by side."""
    df = csv("appendix_methods.csv")
    if df is None:
        return
    piv = df.pivot_table(index="method", columns="split", values="auprc", sort=False)
    piv = piv.sort_values("S1", ascending=True)
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(piv) + 1.2))
    for k, (split, col) in enumerate((("S1", PALETTE[0]), ("S2", PALETTE[1]))):
        if split in piv:
            v = piv[split].values
            ax.barh(y + (0.2 if k == 0 else -0.2), v, height=0.38, color=col,
                    label="S1 random split" if k == 0 else "S2 zero-shot (neurodegenerative)")
            for yi, vi in zip(y + (0.2 if k == 0 else -0.2), v):
                ax.text(vi + 0.008, yi, f"{vi:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y, piv.index, fontsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUPRC (measured, seed 0)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "figA1_methods_s1_s2.png")


def figA2():
    """Seed check: S2 AUPRC per seed for BRIDGE-DR and TransE."""
    df = csv("seed_check_s2.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for k, (m, g) in enumerate(df.groupby("method", sort=False)):
        ax.scatter([k] * len(g), g["auprc"], s=60, color=PALETTE[k], zorder=3)
        for _, r in g.iterrows():
            ax.annotate(f"seed {int(r['seed'])}: {r['auprc']:.3f}", (k, r["auprc"]),
                        xytext=(10, 8 if r["seed"] == 0 else -8), textcoords="offset points", va="center", fontsize=8)
        ax.errorbar(k - 0.12, g["auprc"].mean(), yerr=g["auprc"].std(), fmt="D", color="#555555", capsize=4, ms=5)
    ax.set_xticks(range(df["method"].nunique()), df["method"].unique())
    ax.set_xlim(-0.5, df["method"].nunique() - 0.2)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("S2 AUPRC (neurodegenerative)")
    ax.grid(axis="x", visible=False)
    save(fig, "figA2_seed_check.png")


def figA3():
    """Zero-shot AUROC and AUPRC of BRIDGE-DR for every held-out disease area."""
    df = csv("appendix_areas.csv")
    if df is None:
        return
    df = df.sort_values("auprc")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(df) + 1.2))
    ax.barh(y + 0.2, df["auroc"], height=0.38, color=PALETTE[0], label="AUROC")
    ax.barh(y - 0.2, df["auprc"], height=0.38, color=PALETTE[2], label="AUPRC")
    for yi, a, b in zip(y, df["auroc"], df["auprc"]):
        ax.text(a + 0.008, yi + 0.2, f"{a:.3f}", va="center", fontsize=7.5)
        ax.text(b + 0.008, yi - 0.2, f"{b:.3f}", va="center", fontsize=7.5)
    ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(y, df["area"], fontsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score on held-out area (S2, seed 0)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "figA3_zero_shot_areas.png")


def figA4():
    """ROC and PR curves of every implemented method on S1 and S2."""
    df = csv("appendix_roc_pr_curves.csv")
    if df is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    methods = list(dict.fromkeys(df["method"]))
    for (i, split), (j, curve) in [((i, s), (j, c)) for i, s in enumerate(("S1", "S2")) for j, c in enumerate(("roc", "pr"))]:
        ax = axes[i, j]
        for k, m in enumerate(methods):
            g = df[(df["method"] == m) & (df["split"] == split) & (df["curve"] == curve)]
            if len(g):
                ax.plot(g["x"], g["y"], color=PALETTE[k % len(PALETTE)], lw=2.2 if m == "BRIDGE-DR" else 1.3,
                        label=m, zorder=3 if m == "BRIDGE-DR" else 2)
        if curve == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
        else:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.set_title(f"{'ROC' if curve == 'roc' else 'Precision–recall'} – {split}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "figA4_roc_pr_all_methods.png")


def figA5():
    """Validation AUPRC per epoch for every implemented method."""
    df = csv("appendix_training_curves.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    methods = list(dict.fromkeys(df["method"]))
    for ax, split in zip(axes, ("S1", "S2")):
        for k, m in enumerate(methods):
            g = df[(df["method"] == m) & (df["split"] == split)]
            if len(g):
                ax.plot(g["epoch"], g["val_auprc"], color=PALETTE[k % len(PALETTE)],
                        lw=2.2 if m == "BRIDGE-DR" else 1.2, label=m)
        ax.set_xlabel("Epoch")
        ax.set_title(f"Validation AUPRC during training – {split}", fontsize=10)
    axes[0].set_ylabel("Validation AUPRC")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "figA5_training_curves_all_methods.png")


def figA6():
    """Accuracy vs training cost for every implemented method."""
    df = csv("appendix_methods.csv")
    if df is None:
        return
    df = df[df["method"] != "LLM-only (qwen2.5:3b)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, split in zip(axes, ("S1", "S2")):
        g = df[df["split"] == split]
        for k, r in enumerate(g.itertuples(index=False)):
            ax.scatter(r.minutes, r.auprc, s=90 if r.method == "BRIDGE-DR" else 55, color=PALETTE[k % len(PALETTE)], zorder=3)
            ax.annotate(r.method, (r.minutes, r.auprc), xytext=(6, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Training time on RTX 3050 4 GB (minutes)")
        ax.set_ylabel("Test AUPRC")
        ax.set_title(f"Accuracy vs training cost – {split}", fontsize=10)
    fig.tight_layout()
    save(fig, "figA6_accuracy_vs_cost_all_methods.png")


def figA7():
    """Score distributions of test indications and negatives (BRIDGE-DR)."""
    df = csv("appendix_score_distribution.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, split in zip(axes, ("S1", "S2")):
        g = df[df["split"] == split]
        bins = np.linspace(g["logit"].min(), g["logit"].max(), 40)
        for k, lab in enumerate(("indication", "random negative", "contraindication")):
            ax.hist(g[g["label"] == lab]["logit"], bins=bins, alpha=0.6, color=PALETTE[k], label=lab)
        ax.set_xlabel("BRIDGE-DR score (logit)")
        ax.set_ylabel("Pairs")
        ax.set_title(f"Score distribution on the test set – {split}", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "figA7_score_distribution.png")


def figA8():
    """Path fidelity of REX paths vs model-agnostic paths vs random edges."""
    df = csv("table12_path_fidelity.csv")
    if df is None:
        return
    labels = {"rex": "REX paths\n(attention × gradient)", "path_only": "Path-only\n(graph order)",
              "random": "Random edges\n(control)"}
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(df))
    ax.bar(x, df["fidelity"], color=[PALETTE[0], PALETTE[3], PALETTE[6]][:len(df)], width=0.6)
    for xi, v in zip(x, df["fidelity"]):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [labels.get(e, e) for e in df["explainer"]], fontsize=8.5)
    ax.set_ylabel("Path fidelity (relative score drop)")
    ax.grid(axis="x", visible=False)
    save(fig, "figA8_path_fidelity.png")


def figA9():
    """Zero-shot AUPRC per held-out area: BRIDGE-DR vs R-GCN."""
    a, r = csv("appendix_areas.csv"), csv("rgcn_zero_shot_areas.csv")
    if a is None or r is None:
        return
    df = a[["area", "auprc"]].merge(r[["area", "auprc"]], on="area", suffixes=("_bridge", "_rgcn"))
    df = df.sort_values("auprc_bridge")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(df) + 1.2))
    ax.barh(y + 0.2, df["auprc_bridge"], height=0.38, color=PALETTE[0], label="BRIDGE-DR")
    ax.barh(y - 0.2, df["auprc_rgcn"], height=0.38, color=PALETTE[6], label="R-GCN")
    for yi, b, c in zip(y, df["auprc_bridge"], df["auprc_rgcn"]):
        ax.text(b + 0.008, yi + 0.2, f"{b:.3f}", va="center", fontsize=7.5)
        ax.text(c + 0.008, yi - 0.2, f"{c:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y, df["area"], fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Zero-shot AUPRC on the held-out area (S2, seed 0)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "figA9_zero_shot_bridge_vs_rgcn.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in (fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig14, fig16, figA1, figA2, figA3, figA4, figA5, figA6, figA7, figA8, figA9):
        f()


if __name__ == "__main__":
    main()
