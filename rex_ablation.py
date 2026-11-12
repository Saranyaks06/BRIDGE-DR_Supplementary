"""REX ablations for Table 12: explanations generated without retrieval, and by the LLM alone.

Reuses the predictions explained in the main run (results/primekg/explanations_{s1,s2}.json): the
same drug-disease pairs, the same KG paths and the same retrieved PubMed abstracts. Only the
generation context changes:

  no_retrieval: the LLM sees the KG paths but no abstracts
  llm_only:     the LLM sees only the drug and disease names

Every generated claim is then checked with the same NLI model against all retrieved abstracts of
that pair; a claim is supported when at least one abstract entails it (p > 0.5). Nothing is removed,
so the support rate is the raw rate and hallucination = 1 - support.

  python scripts/rex_ablation.py --out results/rex_ablation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge_dr.rex import EUTILS, OLLAMA, _get  # noqa: E402

PROMPTS = {
    "no_retrieval": ("You explain a predicted drug repurposing candidate: {drug} for {disease}.\n"
                     "Knowledge-graph paths:\n{paths}\n\n"
                     "Write 3-5 short mechanistic claims, one per line, nothing else."),
    "llm_only": ("You explain a predicted drug repurposing candidate: {drug} for {disease}.\n"
                 "Write 3-5 short mechanistic claims, one per line, nothing else."),
}


def abstracts(pmids):
    if not pmids:
        return []
    xml = _get(f"{EUTILS}/efetch.fcgi?db=pubmed&retmode=xml&id={','.join(pmids)}")
    docs = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        text = (art.findtext(".//ArticleTitle") or "") + " " + " ".join("".join(t.itertext()) for t in art.iter("AbstractText"))
        if text.strip():
            docs.append(text)
    return docs


def generate(model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2},
                       "keep_alive": 0}).encode()
    text = json.loads(_get(f"{OLLAMA}/api/generate", body, timeout=600))["response"]
    claims = [re.sub(r"^[\s\-*\d.)]+", "", line).strip() for line in text.splitlines()]
    return [c for c in claims if len(c.split()) >= 4][:5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/primekg")
    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--out", default="results/rex_ablation")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import pipeline
    nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small", device=-1, top_k=None)

    def entailed(claim, docs):
        best = 0.0
        for d in docs:
            res = nli({"text": d[:2000], "text_pair": claim})
            res = res[0] if isinstance(res[0], list) else res
            best = max(best, next((r["score"] for r in res if r["label"].lower().startswith("entail")), 0.0))
        return best > 0.5

    rows, t0 = [], time.time()
    for split in ("s1", "s2"):
        path = os.path.join(a.results, f"explanations_{split}.json")
        if not os.path.exists(path):
            continue
        for e in json.load(open(path, encoding="utf-8")):
            docs = abstracts(e["retrieved_pmids"])
            time.sleep(0.4)
            paths = "\n".join(" -> ".join(q["nodes"]) for q in e["paths"]) or "(none found)"
            for mode, tmpl in PROMPTS.items():
                claims = generate(a.llm, tmpl.format(drug=e["drug"], disease=e["disease"], paths=paths))
                sup = [entailed(c, docs) for c in claims]
                rows.append({"split": split, "mode": mode, "drug": e["drug"], "disease": e["disease"],
                             "n_docs": len(docs), "claims": claims, "supported": sup})
                print(f"[{split}] {mode:12s} {e['drug']} -> {e['disease']}: {sum(sup)}/{len(claims)} supported "
                      f"({len(docs)} abstracts, {time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for mode in PROMPTS:
        r = [x for x in rows if x["mode"] == mode]
        n, s = sum(len(x["claims"]) for x in r), sum(sum(x["supported"]) for x in r)
        summary[mode] = {"explanations": len(r), "claims": n, "supported": s,
                         "claim_support_rate": s / n if n else 0.0, "hallucination_rate": 1 - s / n if n else 0.0}
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows, "llm": a.llm}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
