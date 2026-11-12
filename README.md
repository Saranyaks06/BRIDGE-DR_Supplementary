# BRIDGE-DR

**B**iomarker-conditioned, **R**etrieval-grounded, **I**nductive **D**rug-repurposing **G**raph **E**ngine.
This is an implementation of a multimodal knowledge-graph transformer with conformal false-discovery control and
LLM explanations, for drug repurposing in Alzheimer's disease and Long COVID.

It runs end to end on a laptop (4 GB GPU, 16 GB RAM). Trained models are included, so results can be
reproduced and new diseases ranked without retraining.

- **Results:** [RESULTS.md](RESULTS.md) and the filled report `reports/BRIDGE-DR_Implementation_Report.docx`
- **How to use:** [USER_GUIDE.md](USER_GUIDE.md)

## Architecture

| Module | What it does | Code |
|---|---|---|
| **TriCAL** | Aligns graph, chemical-structure and transcriptome views of each drug with contrastive (InfoNCE) learning, modality dropout and gated fusion; tolerates missing views | `bridge_dr/models/tricals.py` |
| **BC-RGT** | Relational graph transformer over PrimeKG. Each disease is conditioned (FiLM) on its expression signature, which enables subtype-specific and zero-shot ranking | `bridge_dr/models/bcrgt.py` |
| **SRCL** | Signature-reversal consistency loss: drugs that reverse a disease signature (−Spearman) should rank higher | `bridge_dr/models/losses.py` |
| **CFS** | Conformal p-values against confirmed negatives plus Benjamini–Hochberg give a shortlist with bounded false-discovery rate | `bridge_dr/cfs.py` |
| **REX** | KG paths (attention × gradient), PubMed retrieval and a local LLM produce cited hypotheses; unsupported claims are removed by NLI verification | `bridge_dr/rex.py` |

Evaluation protocols: **S1** random edge split, **S2** zero-shot hold-out of a whole disease area,
**S3** time-sliced (needs trial dates).

## Results (PrimeKG, 1 seed, RTX 3050 4 GB)

| | AUROC | AUPRC | CFS empirical FDR at target 0.05 / 0.10 / 0.20 |
|---|---|---|---|
| S1 transductive | 0.942 | 0.908 | 0.045 / 0.073 / 0.164 |
| S2 zero-shot (neurodegenerative) | 0.713 | 0.527 | no candidates pass |

Full loop runtime is 31 minutes. See [RESULTS.md](RESULTS.md) for the per-area, ablation, candidate and explanation results.

## Getting started (5 steps)

**You need:** Python 3.10 or 3.11, Git, about 3 GB of free disk and an internet connection. An NVIDIA GPU with 4 GB or more is optional; everything also runs on CPU, only slower.

**1. Download the code.** Accept the GitHub invitation, then:

```bash
git clone https://github.com/jeevatechoffice500k/bridge-dr.git
cd bridge-dr
```

(Or on the GitHub page: **Code → Download ZIP**, then unzip and open a terminal in the folder.)

**2. Install the packages** (a virtual environment is recommended):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install the matching PyTorch build from https://pytorch.org first (for example the CUDA 11.8 wheel).

**3. Check that it runs (under a minute on a GPU, a few minutes on CPU; no downloads):**

```bash
python scripts/run.py --data synthetic --epochs 150 --patience 100 --explain-n 0 --out results/smoke
```

It trains on a small built-in synthetic graph and should end with a line like `TEST {"auroc": 0.91, "auprc": 0.89, ...}`.

**4. Reproduce the main results with the trained models (no training).** The first run downloads PrimeKG (~1 GB) into `data/primekg/`:

```bash
python scripts/run.py --data primekg --smiles-csv data/primekg/drug_smiles.csv --splits s1 s2 --area neurodegenerative --resume --checkpoint-dir checkpoints --case-disease "Alzheimer disease" "long COVID-19" --explain-n 0 --out results/check
```

Compare `results/check/summary.md` with [RESULTS.md](RESULTS.md). Expect S1 AUROC ≈ 0.94 and AUPRC ≈ 0.91, and S2 AUROC ≈ 0.71 and AUPRC ≈ 0.53. The ranking metrics (MRR, Hits@k, median rank) match exactly. AUPRC can differ by about 0.01 because the random negative pairs are redrawn.

**5. Optional: retrain or regenerate the report.**

```bash
# Retrain the full loop (~30 min on a 4 GB GPU)
python scripts/run.py --data primekg --smiles-csv data/primekg/drug_smiles.csv --splits s1 s2 --area neurodegenerative --case-disease "Alzheimer disease" "long COVID-19" --out results/my_run

# Rebuild the report and supplementary from the saved results (needs the plan .docx in docs/)
bash scripts/finalize.sh
```

For LLM explanations (REX), install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:3b`; without it REX falls back to KG-path templates. See [USER_GUIDE.md](USER_GUIDE.md) for every option and for troubleshooting.

## Project layout

```
bridge_dr/            framework package
  data/               PrimeKG loader, synthetic data, splits S1–S3
  models/             TriCAL, BC-RGT, losses (focal, SRCL)
  trainer.py          training, evaluation, ranking, saliency
  cfs.py  rex.py  metrics.py
scripts/              run.py, run_experiments.py, fetch_smiles.py, export_report_data.py, make_figures.py, fill_report.py
checkpoints/          trained models (S1 and S2)
results/              outputs of every run
reports/              filled report (.docx), figures (.png), editable figure data (.csv)
```

## Scope of this implementation

This build follows the plan's architecture at laptop scale:
- hidden size 128, 2 layers, 1 seed
- edge sampling instead of neighbour sampling, and no Optuna tuning
- Morgan fingerprints instead of ChemBERTa
- disease signatures derived from the knowledge graph

The following plug in through CSV inputs: L1000 (which also switches on SRCL), GEO disease signatures and subtypes, and AACT trial dates for S3.

Baselines reimplemented and run on the same splits: R-GCN, CompGCN, HGT, TransE, DistMult, RotatE and LLM-only ranking (`python scripts/run_experiments.py --only baselines`).
Not run (too large for a 4 GB GPU, or data not supplied): TxGNN, NBFNet, CMap reversal, the S3 time split, subtypes, MedCPT dense retrieval, docking, trained ADMET models and expert scoring; the report leaves them out and lists them in Appendix A.10. Rule-based drug-likeness and BBB checks and ClinicalTrials.gov trial status are included (`scripts/check_candidates.py`).
