"""BRIDGE-DR end-to-end loop: data -> TriCAL + BC-RGT (+SRCL) -> evaluation -> CFS -> case study -> REX.

Examples
  python scripts/run.py --data synthetic --out results/synthetic
  python scripts/run.py --data primekg --smiles-csv data/primekg/drug_smiles.csv \
      --splits s1 s2 --area neurodegenerative --case-disease "alzheimer disease" --out results/primekg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge_dr.cfs import cfs_select, fdr_report, fit_temperature  # noqa: E402
from bridge_dr.data import load_primekg, make_split, make_synthetic  # noqa: E402
from bridge_dr.metrics import calibration  # noqa: E402
from bridge_dr.rex import Explainer  # noqa: E402
from bridge_dr.trainer import TrainConfig, Trainer  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "primekg"], default="synthetic")
    p.add_argument("--kg-path", default="data/primekg/kg.csv")
    p.add_argument("--max-edges-per-relation", type=int, default=150_000)
    p.add_argument("--smiles-csv")
    p.add_argument("--l1000-csv")
    p.add_argument("--disease-sig-csv")
    p.add_argument("--trial-dates-csv")
    p.add_argument("--subtype-csv", help="rows: subtype,<signature genes...> for subtype-specific ranking")
    p.add_argument("--splits", nargs="+", default=["s1"], choices=["s1", "s2", "s3"])
    p.add_argument("--area", default=None, help="disease area held out in S2")
    p.add_argument("--cutoff-year", type=int, default=2020)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-edges", type=int, default=300_000)
    p.add_argument("--eval-max-edges", type=int, default=800_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--case-disease", nargs="*", default=[], help="name substrings of case-study diseases")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--explain-n", type=int, default=5)
    p.add_argument("--llm", default="qwen2.5:3b", help="Ollama model for REX")
    p.add_argument("--no-pubmed", action="store_true")
    p.add_argument("--no-nli", action="store_true")
    # Ablations
    p.add_argument("--no-tricals", action="store_true")
    p.add_argument("--no-film", action="store_true")
    p.add_argument("--no-srcl", action="store_true")
    p.add_argument("--no-modality-dropout", action="store_true")
    p.add_argument("--random-negatives-only", action="store_true")
    p.add_argument("--encoder", default="rgt", choices=["rgt", "rgcn", "compgcn", "hgt", "transe", "distmult", "rotate"],
                   help="rgt = BRIDGE-DR; others are comparison baselines (run without TriCAL and FiLM)")
    p.add_argument("--resume", action="store_true", help="load saved model checkpoints instead of training")
    p.add_argument("--checkpoint-dir", help="folder with model_<split>_seed<k>.pt (default: --out)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/run")
    return p.parse_args()


def find_disease(data, query: str) -> int | None:
    q = query.lower()
    names = [(int(c), data.node_names[c]) for c in data.disease_nodes.tolist()]
    exact = [c for c, n in names if n.lower() == q]
    if exact:
        return exact[0]
    hits = [c for c, n in names if q in n.lower()]
    if not hits:
        return None
    counts = {c: int((data.indication[:, 1] == c).sum()) for c in hits}
    return max(hits, key=counts.get)


def _load_previous(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, "results.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_one(args, data, kind: str, seed: int, out_dir: str, log) -> dict:
    split = make_split(data, kind, seed, area=args.area, cutoff_year=args.cutoff_year)
    log(f"[{kind} seed {seed}] train {len(split.train_pos)} val {len(split.val_pos)} test {len(split.test_pos)} "
        f"contra train {len(split.train_contra)} calib {len(split.calib_null)} test-null {len(split.test_null)}")
    cfg = TrainConfig(
        dim=args.dim, heads=args.heads, layers=args.layers, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, eval_every=args.eval_every, patience=args.patience, max_edges=args.max_edges, eval_max_edges=args.eval_max_edges,
        use_tricals=not args.no_tricals and args.encoder == "rgt", use_film=not args.no_film and args.encoder == "rgt", use_srcl=not args.no_srcl,
        modality_dropout=0.0 if args.no_modality_dropout else 0.3, hard_negatives=not args.random_negatives_only,
        encoder=args.encoder,
    )
    tr = Trainer(data, split, cfg, args.device, seed)
    log(f"  encoder: {cfg.encoder} | SRCL active: {tr.srcl_on} | TriCAL: {cfg.use_tricals} | FiLM: {cfg.use_film} | device {args.device}")
    ckpt = os.path.join(args.checkpoint_dir or out_dir, f"model_{kind}_seed{seed}.pt")
    if args.resume and os.path.exists(ckpt):
        tr.model.load_state_dict(torch.load(ckpt, map_location=args.device))
        old = next((r for r in args.previous if r["split"] == kind and r["seed"] == seed), None)
        fit = {**(old["fit"] if old else {}), "history": []}
        log(f"  loaded checkpoint {ckpt}")
    else:
        fit = tr.fit(split.val_pos, log)
        torch.save(tr.model.state_dict(), ckpt)

    h, ei, et = tr.embed()
    attention = tr.model.last_attention.cpu() if tr.model.last_attention is not None else torch.ones(ei.shape[1])
    res = {"split": kind, "seed": seed, "fit": {k: v for k, v in fit.items() if k != "history"}}
    res["test"] = {**tr.eval_pairs(split.test_pos, h, split.test_null), **tr.rank_metrics(split.test_pos, h)}
    log(f"  TEST {json.dumps({k: round(v, 4) for k, v in res['test'].items()})}")

    # Calibration (temperature scaling fitted on validation)
    with torch.no_grad():
        vneg = tr.corrupt(split.val_pos, tr.all_pos_keys)
        vl = torch.cat([tr.score(h, split.val_pos), tr.score(h, vneg)]).cpu()
        tneg = tr.corrupt(split.test_pos, tr.all_pos_keys)
        tl = torch.cat([tr.score(h, split.test_pos), tr.score(h, tneg)]).cpu()
    vy = torch.cat([torch.ones(len(split.val_pos)), torch.zeros(len(vneg))])
    ty = np.r_[np.ones(len(split.test_pos)), np.zeros(len(tneg))]
    T = fit_temperature(vl, vy)
    res["calibration"] = {
        "temperature": T,
        "before": calibration(ty, torch.sigmoid(tl).numpy()),
        "after": calibration(ty, torch.sigmoid(tl / T).numpy()),
    }

    # CFS: calibrate on held-out confirmed negatives + random nulls, test on fresh nulls + test positives
    area = lambda pairs: np.array([data.disease_area.get(int(c), "other") for c in pairs[:, 1]])  # noqa: E731
    with torch.no_grad():
        # Plan: calibrate only on confirmed negatives (contraindications / repoDB failures);
        # random pairs may be unlabelled positives and would break exchangeability.
        cal_pairs, test_null = split.calib_null, split.test_null
        cal_s = tr.score(h, cal_pairs).cpu().numpy()
        cand = torch.cat([split.test_pos, test_null])
        cand_s = tr.score(h, cand).cpu().numpy()
    is_pos = np.r_[np.ones(len(split.test_pos), bool), np.zeros(len(test_null), bool)]
    res["cfs"] = []
    for a in args.alphas:
        _, sel = cfs_select(cand_s, area(cand), cal_s, area(cal_pairs), a)
        res["cfs"].append({"target_fdr": a, **fdr_report(sel, is_pos)})
    # Comparison: fixed top-k with the same size as CFS at the middle alpha
    k = res["cfs"][len(args.alphas) // 2]["shortlist_size"]
    topk = np.zeros(len(cand_s), bool)
    topk[np.argsort(-cand_s)[:k]] = True
    res["cfs_topk_baseline"] = fdr_report(topk, is_pos)
    log(f"  CFS {res['cfs']}  | top-k {res['cfs_topk_baseline']}")
    # Raw scores for figures (PR curves, reliability diagrams, FDR plots, embedding maps)
    np.savez_compressed(
        os.path.join(out_dir, f"scores_{kind}_seed{seed}.npz"),
        test_y=ty, test_logit=tl.numpy(), temperature=T, cand_score=cand_s, cand_is_pos=is_pos,
        cal_score=cal_s, drug_emb=h[tr.drug_nodes].float().cpu().numpy(),
    )

    # Case studies + REX (first seed only)
    if seed == args.seeds[0] and args.case_disease:
        cases, explain_pairs = [], []
        known = set(map(tuple, torch.cat([split.train_pos, split.val_pos]).tolist()))
        a_mid = args.alphas[len(args.alphas) // 2]
        for q in args.case_disease:
            c = find_disease(data, q)
            if c is None:
                log(f"  case '{q}': disease not found")
                continue
            s = tr.drug_scores(h, c).cpu().numpy()
            dn = data.drug_nodes.numpy()
            g = data.disease_area.get(c, "other")
            p, sel = cfs_select(s, np.full(len(s), g), cal_s, area(cal_pairs), a_mid)
            order = [i for i in np.argsort(-s) if (int(dn[i]), c) not in known][: args.top_k]
            rows = [{"rank": r + 1, "drug": data.node_names[dn[i]], "disease": data.node_names[c],
                     "score": float(s[i]), "conformal_p": float(p[i]), f"in_cfs_shortlist@{a_mid}": bool(sel[i]),
                     "known_test_positive": (int(dn[i]), c) in set(map(tuple, split.test_pos.tolist()))}
                    for r, i in enumerate(order)]
            cases.append({"query": q, "disease": data.node_names[c], "cfs_shortlist_size": int(sel.sum()), "top": rows})
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"candidates_{kind}_{q.replace(' ', '_')}.csv"), index=False)
            log(f"  case '{data.node_names[c]}': CFS shortlist {int(sel.sum())} drugs; top-3 "
                f"{[r['drug'] for r in rows[:3]]}")
            explain_pairs += [(int(dn[i]), c) for i in order[: args.explain_n]]

            if args.subtype_csv:
                sub = pd.read_csv(args.subtype_csv)
                gene_ix = {gname: j for j, gname in enumerate(data.sig_genes)}
                tops = {}
                for row in sub.itertuples(index=False):
                    sig = torch.zeros(len(data.sig_genes))
                    for gname, v in zip(sub.columns[1:], row[1:]):
                        if gname in gene_ix:
                            sig[gene_ix[gname]] = float(v)
                    ss = tr.drug_scores(h, c, sig.unsqueeze(0)).cpu().numpy()
                    tops[str(row[0])] = set(np.argsort(-ss)[:50].tolist())
                names = list(tops)
                cases[-1]["subtype_top50_jaccard"] = {
                    f"{a}|{b}": len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                    for i, a in enumerate(names) for b in names[i + 1:]}
        res["cases"] = cases

        if explain_pairs and args.explain_n > 0:
            log(f"  REX: explaining {len(explain_pairs)} predictions ...")
            sal = tr.saliency(torch.tensor(explain_pairs))
            full_ei, full_et = tr.graph(None)
            rex = Explainer(data, full_ei.cpu(), full_et.cpu(), attention, sal, llm_model=args.llm, att_edge_index=ei.cpu(),
                            use_pubmed=not args.no_pubmed, rel_names=tr.rel_names,
                            nli_model=None if args.no_nli else "cross-encoder/nli-deberta-v3-small")
            exps = []
            for d, c in explain_pairs:
                e = rex.explain(d, c)
                exps.append(e)
                log(f"    {e['drug']} -> {e['disease']}: {len(e['paths'])} paths, {len(e['retrieved_pmids'])} PMIDs, "
                    f"{len(e['claims'])} supported claims, {e['removed_claims']} removed ({e['generator']})")
                time.sleep(0.4)  # stay under NCBI's 3 requests/s
            res["rex"] = {
                "n": len(exps),
                "mean_claim_support_rate": float(np.mean([e["claim_support_rate"] for e in exps])),
                "total_removed_claims": int(sum(e["removed_claims"] for e in exps)),
            }
            with open(os.path.join(out_dir, f"explanations_{kind}.json"), "w", encoding="utf-8") as f:
                json.dump(exps, f, indent=2, ensure_ascii=False)
            del rex

    del tr, h
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    with open(os.path.join(out_dir, f"history_{kind}_seed{seed}.json"), "w") as f:
        json.dump(fit["history"], f, indent=2)
    return res


def summarize(results: list[dict]) -> str:
    lines = ["# BRIDGE-DR results", ""]
    for kind in dict.fromkeys(r["split"] for r in results):
        rs = [r for r in results if r["split"] == kind]
        lines += [f"## Split {kind.upper()} ({len(rs)} seed{'s' if len(rs) > 1 else ''})", "",
                  "| Metric | Value |", "|---|---|"]
        for k in rs[0]["test"]:
            v = np.array([r["test"][k] for r in rs])
            lines.append(f"| {k} | {v.mean():.4f}" + (f" ± {v.std():.4f}" if len(v) > 1 else "") + " |")
        c = rs[0]["calibration"]
        lines += ["", f"Calibration (T = {c['temperature']:.3f}): ECE {c['before']['ece']:.4f} -> {c['after']['ece']:.4f}, "
                  f"Brier {c['before']['brier']:.4f} -> {c['after']['brier']:.4f}", "",
                  "| Selection | Target FDR | Empirical FDR | Power | Shortlist |", "|---|---|---|---|---|"]
        for x in rs[0]["cfs"]:
            lines.append(f"| CFS | {x['target_fdr']} | {x['empirical_fdr']:.3f} | {x['power']:.3f} | {x['shortlist_size']} |")
        t = rs[0]["cfs_topk_baseline"]
        lines.append(f"| Top-k (same size) | none | {t['empirical_fdr']:.3f} | {t['power']:.3f} | {t['shortlist_size']} |")
        f = rs[0]["fit"]
        if "train_seconds" in f:
            lines += ["", f"Training: {f['train_seconds'] / 60:.1f} min, peak VRAM {f['peak_vram_gb']:.2f} GB", ""]
        for case in rs[0].get("cases", []):
            lines += [f"### Candidates: {case['disease']} (CFS shortlist {case['cfs_shortlist_size']} drugs)", "",
                      "| Rank | Drug | Score | Conformal p | In shortlist |", "|---|---|---|---|---|"]
            for row in case["top"]:
                in_sl = next(v for k, v in row.items() if k.startswith("in_cfs_shortlist"))
                lines.append(f"| {row['rank']} | {row['drug']} | {row['score']:.3f} | {row['conformal_p']:.4f} | {in_sl} |")
            lines.append("")
        if "rex" in rs[0]:
            x = rs[0]["rex"]
            lines += [f"REX: {x['n']} explanations, mean claim support rate {x['mean_claim_support_rate']:.2f}, "
                      f"{x['total_removed_claims']} unsupported claims removed", ""]
    return "\n".join(lines)


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"=== BRIDGE-DR {time.strftime('%Y-%m-%d %H:%M')} args: {vars(args)}")
    t0 = time.time()
    if args.data == "synthetic":
        data = make_synthetic()
        if "s2" in args.splits and args.area is None:
            args.area = "area_0"
    else:
        data = load_primekg(args.kg_path, args.max_edges_per_relation, smiles_csv=args.smiles_csv,
                            l1000_csv=args.l1000_csv, disease_sig_csv=args.disease_sig_csv,
                            trial_dates_csv=args.trial_dates_csv)
    log(f"Loaded {args.data} in {time.time() - t0:.0f}s: {data.summary()}")

    args.previous = _load_previous(args.out)  # read before results.json is overwritten
    results = []
    for kind in args.splits:
        for seed in args.seeds:
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            results.append(run_one(args, data, kind, seed, args.out, log))
            with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summarize(results))
    log(f"Done in {(time.time() - t0) / 60:.1f} min. Results in {args.out}")


if __name__ == "__main__":
    main()
