"""Rule-based ADMET/BBB checks and ClinicalTrials.gov status for the reported top-10 candidates.

Reads reports/figure_data/table15_candidates_ad.csv and table16_candidates_long_covid.csv, looks up
each drug's SMILES (data/primekg/drug_smiles.csv via the DrugBank ids in kg.csv), computes RDKit
descriptors and queries the ClinicalTrials.gov v2 API. Writes
reports/figure_data/table15_16_candidate_checks.csv, which fill_report.py and make_figures.py read.

Rules (screening heuristics, not predictions from a trained ADMET model):
  Lipinski: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10; drug-like when at most 1 rule is broken.
  BBB+:     TPSA < 90, MW < 450, HBD <= 3 (CNS-penetration rule of thumb).

  python scripts/check_candidates.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reports", "figure_data")
CT_URL = "https://clinicaltrials.gov/api/v2/studies"
CONDITION = {"Alzheimer disease": "Alzheimer disease", "long COVID-19": "Post-Acute COVID-19 Syndrome"}
PHASE_ORDER = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]


def drug_ids(names):
    ids = {}
    for chunk in pd.read_csv(os.path.join(ROOT, "data", "primekg", "kg.csv"), usecols=["x_type", "x_id", "x_name"],
                             chunksize=1_000_000, low_memory=False):
        d = chunk[(chunk["x_type"] == "drug") & chunk["x_name"].isin(names)]
        ids.update(zip(d["x_name"], d["x_id"]))
        if len(ids) == len(names):
            break
    return ids


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {}
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba, tpsa = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), rdMolDescriptors.CalcTPSA(mol)
    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return {"mw": round(mw, 1), "logp": round(logp, 2), "hbd": hbd, "hba": hba, "tpsa": round(tpsa, 1),
            "lipinski_violations": violations, "drug_like": violations <= 1,
            "bbb_rule": tpsa < 90 and mw < 450 and hbd <= 3}


def trials(drug, disease):
    q = urllib.parse.urlencode({"query.cond": CONDITION[disease], "query.intr": drug, "countTotal": "true",
                                "pageSize": 100, "fields": "protocolSection.designModule.phases,"
                                                           "protocolSection.statusModule.overallStatus"})
    with urllib.request.urlopen(f"{CT_URL}?{q}", timeout=60) as r:
        res = json.loads(r.read())
    phases = [p for s in res.get("studies", []) for p in s.get("protocolSection", {}).get("designModule", {}).get("phases", [])]
    best = max(phases, key=lambda p: PHASE_ORDER.index(p) if p in PHASE_ORDER else -1, default=None)
    n = res.get("totalCount", len(res.get("studies", [])))
    status = "no registered trial" if n == 0 else f"{n} trial{'s' if n > 1 else ''}" + (
        f", up to {best.replace('EARLY_', 'early ').replace('PHASE', 'phase ').lower()}" if best and best in PHASE_ORDER else "")
    return n, best or "", status


def main():
    cands = pd.concat([pd.read_csv(os.path.join(DATA, f)) for f in
                       ("table15_candidates_ad.csv", "table16_candidates_long_covid.csv")], ignore_index=True)
    ids = drug_ids(set(cands["drug"]))
    smiles = pd.read_csv(os.path.join(ROOT, "data", "primekg", "drug_smiles.csv")).set_index("drugbank_id")["smiles"]
    rows = []
    for r in cands.itertuples(index=False):
        sm = smiles.get(ids.get(r.drug))
        n, phase, status = trials(r.drug, r.disease)
        time.sleep(0.3)
        rows.append({"disease": r.disease, "rank": r.rank, "drug": r.drug, "drugbank_id": ids.get(r.drug, ""),
                     "has_structure": isinstance(sm, str), **descriptors(sm),
                     "n_trials": n, "max_phase": phase, "trial_status": status})
        print(f"{r.disease[:12]:12s} {r.drug:24s} {status}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "table15_16_candidate_checks.csv"), index=False)
    print(f"Saved {len(out)} rows to reports/figure_data/table15_16_candidate_checks.csv")


if __name__ == "__main__":
    main()
