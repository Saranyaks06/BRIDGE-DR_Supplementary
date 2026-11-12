# BRIDGE-DR User Guide

This guide covers installing the project, re-using the trained models, re-running experiments,
predicting candidates for any disease, and regenerating the report and its figures.

---

## 1. What is in the project

| Folder / file | Contents |
|---|---|
| `bridge_dr/` | The framework: data loading and splits, TriCAL, BC-RGT, SRCL, CFS, REX, metrics |
| `scripts/run.py` | One full loop: train (or load) → evaluate → CFS → candidates → REX |
| `scripts/run_experiments.py` | The full experiment set used in the report (main run, 9 disease areas, ablations) |
| `scripts/fetch_smiles.py` | Downloads drug structures (SMILES) from PubChem |
| `scripts/export_report_data.py` | Collects results into editable CSVs (`reports/figure_data/`) |
| `scripts/make_figures.py` | Draws all figures from those CSVs (`reports/figures/`) |
| `scripts/fill_report.py` | Writes the filled implementation document (`reports/BRIDGE-DR_Implementation_Report.docx`) |
| `checkpoints/` | Trained models: `model_s1_seed0.pt` (S1, all diseases), `model_s2_seed0.pt` (S2, neurodegenerative held out) |
| `results/` | Raw outputs of every run (`results.json`, `summary.md`, candidates, explanations, logs) |
| `RESULTS.md` | Short written summary of the results |

---

## 2. Installation (Windows, macOS or Linux)

**Requirements:** Python 3.10+, about 3 GB free disk, and an NVIDIA GPU with ≥ 4 GB (optional; the CPU works but is slower).

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version from https://pytorch.org (for example the CUDA 11.8 wheel).

**Optional, for LLM explanations (REX):**
1. Install Ollama from https://ollama.com.
2. Open a **new** terminal and download the model (~2 GB):
   ```bash
   ollama pull qwen2.5:3b
   ```
   If `ollama` is "not recognized", the terminal was opened before installation. Open a new one, or use the full path
   `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` on Windows.
3. Ollama runs in the background on `localhost:11434`; nothing else to configure.
   Without Ollama, REX still runs and writes template explanations from graph paths.

The first REX run downloads the NLI verification model (~200 MB) automatically.

---

## 3. Get the data

PrimeKG (~1 GB) is downloaded automatically on the first run to `data/primekg/kg.csv`.
Download drug structures once (about 10 minutes):

```bash
python scripts/fetch_smiles.py --out data/primekg/drug_smiles.csv
```

---

## 4. Use the trained models (no training needed)

The models in `checkpoints/` are re-used with `--resume --checkpoint-dir checkpoints`.
Training is skipped; evaluation, CFS, candidate ranking and REX run in a few minutes.

**Rank drugs for any disease** (the name must match a PrimeKG disease name; partial names work):
```bash
python scripts/run.py --data primekg --smiles-csv data/primekg/drug_smiles.csv --splits s1 --resume --checkpoint-dir checkpoints --case-disease "Parkinson disease" --top-k 20 --explain-n 5 --out results/parkinson
```

Outputs in `results/parkinson/`:
- `candidates_s1_Parkinson_disease.csv`: ranked drugs with score, conformal p-value and CFS shortlist flag
- `explanations_s1.json`: REX paths, PubMed IDs and verified claims for the top `--explain-n` drugs
- `summary.md`: all metrics in tables

**Use the zero-shot model** (trained without any neurodegenerative indications) by changing the split:
```bash
python scripts/run.py --data primekg --smiles-csv data/primekg/drug_smiles.csv --splits s2 --area neurodegenerative --resume --checkpoint-dir checkpoints --case-disease "Alzheimer disease" --out results/ad_zero_shot
```

> The checkpoints only match the graph they were trained on. Keep the default `--max-edges-per-relation 150000`
> and the same `kg.csv`; otherwise retrain.

---

## 5. Re-run training

**One full loop** (S1 + S2, AD and long COVID case studies, REX), about 30 minutes on a 4 GB laptop GPU:
```bash
python scripts/run.py --data primekg --smiles-csv data/primekg/drug_smiles.csv --splits s1 s2 --area neurodegenerative --epochs 60 --case-disease "Alzheimer disease" "long COVID-19" --explain-n 5 --out results/my_run
```

**The full experiment set from the report** (main run, the other 8 disease areas, 4 ablations), about 3 hours.
Finished experiments are skipped, so it can be stopped and restarted:
```bash
python scripts/run_experiments.py
```

**Quick check without downloads** (synthetic data, about 1 minute):
```bash
python scripts/run.py --data synthetic --splits s1 s2 --epochs 150 --batch-size 128 --case-disease disease_3 --no-pubmed --out results/synthetic
```

### Useful options of `scripts/run.py`

| Option | Meaning | Default |
|---|---|---|
| `--splits s1 s2 s3` | Evaluation protocols to run | `s1` |
| `--area NAME` | Disease area held out in S2: `neurodegenerative`, `mental_health`, `cardiovascular`, `metabolic_disorder`, `autoimmune`, `cell_proliferation`, `anemia`, `adrenal_gland`, `diabetes` | – |
| `--seeds 0 1` | Random seeds (results are averaged as mean ± SD) | `0` |
| `--epochs`, `--dim`, `--layers`, `--heads` | Model size and training length | 60, 128, 2, 4 |
| `--max-edges` | Edges sampled per training step (lower it if the GPU runs out of memory) | 300000 |
| `--case-disease "A" "B"` | Diseases to rank drugs for | – |
| `--top-k`, `--explain-n` | Candidates listed / explained by REX | 10, 5 |
| `--llm` | Ollama model for REX | `qwen2.5:3b` |
| `--no-pubmed`, `--no-nli` | Turn off PubMed retrieval / NLI verification | off |
| `--no-tricals`, `--no-film`, `--no-srcl`, `--no-modality-dropout`, `--random-negatives-only` | Ablations A1, A3, A4, A2, A6 | off |
| `--resume --checkpoint-dir DIR` | Load trained models instead of training | off |
| `--device cpu` | Force CPU | GPU if available |

### Adding the optional datasets from the plan

| Data | Flag | CSV columns | Effect |
|---|---|---|---|
| LINCS L1000 consensus signatures | `--l1000-csv` | `drugbank_id,<gene symbols…>` | Adds the transcriptome view to TriCAL and switches on SRCL |
| GEO disease signatures (e.g. limma t-statistics) | `--disease-sig-csv` | `disease_name,<gene symbols…>` | Replaces KG-derived signatures |
| Subtype signatures (e.g. NMF subtypes) | `--subtype-csv` | `subtype,<gene symbols…>` | Ranks drugs per subtype for each case disease |
| AACT trial start dates | `--trial-dates-csv` | `drugbank_id,disease_name,year` | Enables the time-sliced split S3 |

---

## 6. Regenerate the report and figures

```bash
python scripts/export_report_data.py
python scripts/make_figures.py
python scripts/fill_report.py
```

**To change a value in a figure or table:**
1. Edit the CSV in `reports/figure_data/`. For example, add a baseline row to `table07_s1_prediction.csv`, or correct a number.
2. Run only the last two commands. `export_report_data.py` would overwrite your edits with the raw results.

The document is rebuilt from the original plan in `docs/`, so the plan file itself is never changed.

| CSV | Used for |
|---|---|
| `table07_s1_prediction.csv`, `fig02_pr_curve_s1.csv` | Table 7, Figure 2 |
| `table08_fig03_zero_shot_areas.csv` | Table 8, Figure 3 |
| `fig04_recall_at_k.csv` | Figure 4 |
| `table11_fig05_fdr.csv`, `fig06_reliability_s1.csv` | Table 11, Figures 5–6 |
| `table12_fig07_explanations.csv` | Table 12, Figure 7 |
| `table13_fig08_09_10_ablations.csv` | Table 13, Figures 8–10 |
| `table14_cost.csv` | Table 14 |
| `table15_candidates_ad.csv`, `table16_candidates_long_covid.csv` | Tables 15–16 |
| `fig11_drug_embedding_tsne.csv`, `fig14_rex_subgraph.csv`, `fig16_funnel.csv` | Figures 11, 14, 16 |
| `table15_16_candidate_checks.csv`, `table16_immune_targets.csv` | BBB+, immune-target and trial-status columns of Tables 13–14, Figure 13, Appendix A.8, Table S17 |
| `seed_check_s2.csv` | Appendix A.4, Table S9b |
| `rgcn_zero_shot_areas.csv` | R-GCN column of Table 8, Appendix A.3 |
| `table12_path_fidelity.csv`, `table14_cost_measured_cpu.csv`, `table14_cost_measured_cuda.csv` | Path fidelity in Table 10; peak RAM and CPU-only columns of Table 12 |
| `appendix_roc_pr_curves.csv`, `appendix_training_curves.csv`, `appendix_score_distribution.csv` | Figures A4, A5, A7 |
| `appendix_methods.csv`, `appendix_areas.csv`, `appendix_calibration.csv`, `table12_rex_ablation.csv` | Appendix A tables and figures, Table 10 rows 3–4 |
| `baselines_s1.csv`, `baselines_zero_shot.csv` | Measured baseline rows in Tables 7–8; all rows in Figures 2–3 |

**Report layout:** sections 1–3 are the plan as written. Section 4 holds only results computed by this implementation: parts that were not run (S3, subtypes, TxGNN, NBFNet, CMap, docking, expert scores) are removed and the remaining sections, tables and figures are renumbered. Yellow marks values from the implementation, green marks headings and captions of implemented parts. Tables A1–A10 and Figures A1–A9 follow with further results, and Section 5 lists the original links of every dataset, model and service (data and code availability). A literature review with 19 verified references can be added back with `python scripts/fill_report.py --with-literature`.

**Other measurement scripts** (rerun when the inputs change): `scripts/target_check.py` (immune/endothelial targets), `scripts/path_fidelity.py` (Table 10 path fidelity), `scripts/rex_ablation.py` (Table 10 rows 3–4), `scripts/cost_check.py --device cpu` and `--device cuda` (Table 12).

**Candidate checks** (needs internet; run again whenever the candidate lists change):

```bash
python scripts/check_candidates.py
```

It computes RDKit drug-likeness and BBB rules and queries the ClinicalTrials.gov API. It is not part of `scripts/finalize.sh` because it needs the network.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `--max-edges` (e.g. 150000) and `--eval-max-edges` (e.g. 400000), or close other GPU programs (Ollama keeps the LLM in GPU memory for a few minutes) |
| Training is much slower than usual | Another program is using the GPU, often Ollama right after REX. Wait a few minutes or restart Ollama |
| `ollama` is not recognized | Open a new terminal after installing Ollama |
| REX says "template" instead of "ollama" | Ollama is not running or the model is not pulled: run `ollama pull qwen2.5:3b` |
| `disease not found` | Use the PrimeKG disease name; the loader matches substrings, case-insensitive |
| PubMed errors | No internet or NCBI rate limits; REX continues without abstracts. Add `--no-pubmed` to skip |
