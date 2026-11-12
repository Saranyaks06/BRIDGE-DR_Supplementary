"""Immune / endothelial target check for the reported candidates (Table 16 column).

A drug counts as having an immune or endothelial target when at least one of its PrimeKG targets
(drug_protein, display relation "target") is linked to a biological process whose name matches
immune, inflammatory, cytokine, leukocyte, lymphocyte, T/B cell, interleukin, interferon,
endothelial, angiogenesis or vascular terms (PrimeKG bioprocess_protein edges, from GO).

Writes reports/figure_data/table16_immune_targets.csv (one row per candidate drug).

  python scripts/target_check.py
"""

from __future__ import annotations

import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reports", "figure_data")
PROCESS = re.compile(r"immun|inflamm|cytokine|leukocyte|lymphocyte|\bT cell|\bB cell|interleukin|interferon|"
                     r"endotheli|angiogen|vascular|vasculogen", re.I)


def main():
    cands = pd.concat([pd.read_csv(os.path.join(DATA, f)) for f in
                       ("table15_candidates_ad.csv", "table16_candidates_long_covid.csv")], ignore_index=True)
    drugs = set(cands["drug"])
    targets, proc = {}, {}
    cols = ["relation", "display_relation", "x_type", "x_name", "y_type", "y_name"]
    for c in pd.read_csv(os.path.join(ROOT, "data", "primekg", "kg.csv"), usecols=cols, chunksize=2_000_000,
                         low_memory=False):
        d = c[(c["relation"] == "drug_protein") & (c["display_relation"] == "target") & (c["x_type"] == "drug")
              & c["x_name"].isin(drugs)]
        for drug, gene in zip(d["x_name"], d["y_name"]):
            targets.setdefault(drug, set()).add(gene)
        b = c[(c["relation"] == "bioprocess_protein") & (c["x_type"] == "gene/protein")]
        b = b[b["y_name"].str.contains(PROCESS, na=False)]
        for gene, p in zip(b["x_name"], b["y_name"]):
            proc.setdefault(gene, set()).add(p)
    rows = []
    for r in cands.itertuples(index=False):
        t = sorted(targets.get(r.drug, set()))
        hit = [g for g in t if g in proc]
        rows.append({"disease": r.disease, "rank": r.rank, "drug": r.drug, "n_targets": len(t),
                     "immune_endothelial_targets": ", ".join(hit),
                     "label": ("yes (" + ", ".join(hit[:3]) + (", …" if len(hit) > 3 else "") + ")") if hit
                     else ("no" if t else "no known target")})
        print(f"{r.disease[:12]:12s} {r.drug:22s} {rows[-1]['label']}")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "table16_immune_targets.csv"), index=False)


if __name__ == "__main__":
    main()
