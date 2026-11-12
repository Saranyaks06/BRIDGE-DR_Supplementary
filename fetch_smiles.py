"""Fetch canonical SMILES for PrimeKG drugs from PubChem (by drug name) -> drugbank_id,smiles CSV.

Only drugs with at least one indication or contraindication are queried by default.
  python scripts/fetch_smiles.py --kg-path data/primekg/kg.csv --out data/primekg/drug_smiles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES,IsomericSMILES,SMILES/JSON"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-path", default="data/primekg/kg.csv")
    p.add_argument("--out", default="data/primekg/drug_smiles.csv")
    p.add_argument("--all-drugs", action="store_true")
    a = p.parse_args()

    df = pd.read_csv(a.kg_path, usecols=["relation", "x_type", "x_id", "x_name"], low_memory=False)
    df = df[df["x_type"] == "drug"]
    if not a.all_drugs:
        df = df[df["relation"].isin(["indication", "contraindication", "off-label use"])]
    drugs = df.drop_duplicates("x_id")[["x_id", "x_name"]].values.tolist()

    done = set()
    if os.path.exists(a.out):
        done = set(pd.read_csv(a.out)["drugbank_id"].astype(str))
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["drugbank_id", "smiles"])
        for i, (did, name) in enumerate(drugs):
            if str(did) in done:
                continue
            smi = ""
            try:
                url = PUBCHEM.format(urllib.parse.quote(str(name)))
                props = json.loads(urllib.request.urlopen(url, timeout=20).read())["PropertyTable"]["Properties"][0]
                smi = props.get("IsomericSMILES") or props.get("CanonicalSMILES") or props.get("SMILES", "")
            except Exception:
                pass
            w.writerow([did, smi])
            if i % 100 == 0:
                print(f"{i}/{len(drugs)}", flush=True)
                f.flush()
            time.sleep(0.22)  # PubChem limit: 5 requests/s


if __name__ == "__main__":
    main()
