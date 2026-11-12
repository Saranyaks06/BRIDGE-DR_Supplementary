# BRIDGE-DR: results of the laptop run

**Setup:** PrimeKG (129,375 nodes; 2.26M structural edges after capping each relation at 150k; 11,831 indications; 30,675 contraindications), Morgan fingerprints for 1,846 drugs (PubChem SMILES), KG-derived disease signatures (512 genes), 1 seed.
**Hardware:** RTX 3050 Laptop 4 GB, 16 GB RAM. **Total runtime:** 31 min (training 7.1 + 6.5 min, peak VRAM 3.3 GB).
**Not used in this run:** L1000 transcriptomes, so SRCL was inactive and TriCAL fused two views (graph + structure).

Full tables: [`results/primekg/summary.md`](results/primekg/summary.md). Raw outputs: `results/primekg/`.

## 1. Prediction

| Split | AUROC | AUPRC | MRR | Hits@10 | Hits@100 | Median rank |
|---|---|---|---|---|---|---|
| S1 transductive (random 80/10/10) | **0.942** | **0.908** | 0.114 | 0.190 | 0.498 | 102 |
| S2 zero-shot (all neurodegenerative indications held out) | 0.713 | 0.527 | 0.006 | 0.009 | 0.069 | 1,190 |

AUROC and AUPRC use 1 random negative plus 1 confirmed negative (contraindication) per positive. Ranks are filtered ranks among all 7,957 drugs.

## 2. Conformal FDR control (CFS)

Calibration uses held-out contraindications only. Empirical FDR is measured on a separate set of held-out contraindications plus the test positives.

| Split | Target FDR | Empirical FDR | Power | Shortlist |
|---|---|---|---|---|
| S1 | 0.05 | **0.045** | 0.48 | 598 |
| S1 | 0.10 | **0.073** | 0.58 | 738 |
| S1 | 0.20 | **0.164** | 0.72 | 1,012 |
| S2 | 0.05–0.20 | 0.000 | 0.00 | 0 |

- **S1:** FDR stays at or below every target.
- **S2:** the zero-shot scores are not separated enough from the confirmed negatives, so CFS correctly returns no candidates.
- **Per-disease shortlists** (AD, long COVID) apply Benjamini–Hochberg over all 7,957 drugs. That is very strict, and no drug passed in this run, even with p-values as low as 0.0007.

**Calibration (ECE):**
- S1: 0.110 → 0.106 after temperature scaling.
- S2: 0.298 → 0.321. Temperature fitted on in-distribution validation data does not transfer to held-out disease areas.

## 3. Case-study candidates (top 5, excluding known training/validation indications)

| Model | Alzheimer disease | long COVID-19 |
|---|---|---|
| S1 | Levomethadone, Demexiptiline, Setiptiline, Cetilistat, Rucaparib | Succinimide, Methylprednisolone, Dexamethasone, Alglucosidase alfa, Prednisolone |
| S2 (zero-shot, AD never seen) | Fendiline, **Ipidacrine**, Linagliptin, Levacetylmethadol, Rucaparib | Succinimide, Filgrastim, Somatotropin, Afamelanotide, Methylprednisolone |

- **Ipidacrine** is a cholinesterase inhibitor used clinically for memory disorders. It was ranked #2 for AD with no neurodegenerative labels in training.
- **Linagliptin** (a DPP-4 inhibitor) has been studied for neuroprotection.
- **Corticosteroids** for long COVID reflect PrimeKG's COVID-19 neighbourhood.
- **Many other candidates** (opioids, tricyclic antidepressants) come from psychiatric and pain links and are not credible AD treatments without further validation.

Full lists: `results/primekg/candidates_*.csv`.

**Candidate checks** (S1 top-10; `scripts/check_candidates.py`, output `reports/figure_data/table15_16_candidate_checks.csv`):

| | Alzheimer disease | long COVID-19 |
|---|---|---|
| Drug-like (Lipinski, ≤ 1 violation) | 6 / 10 | 8 / 10 |
| BBB+ (TPSA < 90, MW < 450, HBD ≤ 3) | 5 / 10 | – |
| Any registered trial for the condition | 1 / 10 (Vitamin E, 9 trials, up to phase 3) | 6 / 10 (corticosteroids, testosterone) |

- **Biologics:** Reteplase, Alteplase and Alglucosidase alfa have no small-molecule structure, so the rules do not apply to them.
- **What these checks are:** drug-likeness and BBB are rules of thumb on PubChem structures, not a trained ADMET model. Trial counts are ClinicalTrials.gov studies that list the drug as an intervention; for long COVID the condition is "Post-Acute COVID-19 Syndrome". A listed trial may use the drug only as a comparator.

## 4. Explanations (REX)

Setup: qwen2.5:3b via Ollama, PubMed retrieval with BM25, DeBERTa-v3 NLI verification.

| | S1 | S2 |
|---|---|---|
| Explanations | 10 | 10 |
| Claim support rate after verification | 0.06 | 0.28 |
| Unsupported claims removed | 31 | 27 |

- **Paths found:** AD predictions get 5 KG paths each, e.g. `Levomethadone → opiate dependence → BDNF → Alzheimer disease`. long COVID-19 has almost no KG links, so no paths were found.
- **Claim quality:** most claims from the 3B model were not entailed by their cited abstract and were removed. A larger model (the plan's 7–8B) is expected to raise the support rate.
- **Template claims:** when no abstract is found (e.g. Fendiline in S2), claims are KG-path templates with no literature citation. They count as KG-supported only.

## 5. Gap to the plan's expected results

The plan's targets for S1 were AUPRC 0.951 and the zero-shot neurodegenerative AUPRC 0.79. This run reached 0.908 and 0.527. Differences in setup:
- hidden size 128 with 2 layers (plan: 256 with 3)
- edge sampling instead of neighbour sampling
- no Optuna tuning, 1 seed
- no L1000 data, so no SRCL
- KG-derived disease signatures instead of GEO
- keyword-based disease areas instead of TxGNN's splits

Sections 6–8 compare against baselines reimplemented in the same code and run on the same splits.
TxGNN, NBFNet and the other published systems were not run; the report reports only the methods that were run ; `python scripts/fill_report.py --with-literature` adds a comparison with published work.

## 6. Baselines (measured, same splits, 1 seed)

| Method | S1 AUROC | S1 AUPRC | S2 AUROC | S2 AUPRC |
|---|---|---|---|---|
| **BRIDGE-DR** | **0.942** | **0.908** | 0.713 | 0.527 |
| HGT | 0.932 | 0.886 | 0.693 | 0.517 |
| R-GCN | 0.931 | 0.888 | 0.638 | 0.403 |
| CompGCN | 0.904 | 0.848 | 0.621 | 0.390 |
| TransE | 0.849 | 0.763 | **0.839** | **0.641** |
| RotatE | 0.648 | 0.475 | 0.710 | 0.491 |
| DistMult | 0.582 | 0.422 | 0.508 | 0.343 |
| LLM-only (qwen2.5:3b) | 0.826 | 0.685 | 0.793 | 0.599 |

- **S1:** BRIDGE-DR is best on every metric.
- **S2 zero-shot:** TransE beats BRIDGE-DR, and a second seed confirms it:

  | S2 neurodegenerative | Seed 0 | Seed 1 | Mean ± SD |
  |---|---|---|---|
  | BRIDGE-DR AUPRC | 0.527 | 0.314 | 0.420 ± 0.150 |
  | TransE AUPRC | 0.641 | 0.625 | 0.633 ± 0.012 |

  BRIDGE-DR's zero-shot result is unstable across seeds; with seed 1 its AUROC is 0.48 (chance level). In both seeds its in-distribution validation AUPRC is about 0.96, so training worked. Early stopping uses validation data from the areas the model trains on, so it cannot select for transfer to the held-out area. Possible fixes, not tested here: validate on a held-out disease area, add stronger regularisation, or use the plan-size model with L1000 and SRCL, which the plan relies on for zero-shot transfer. TransE's indication and contraindication translation vectors appear to transfer more reliably.
- **LLM-only** also beats BRIDGE-DR on S2 (AUPRC 0.599 vs 0.527). The test pairs are mostly approved indications, which the LLM may already know from pretraining, so this is not a clean zero-shot comparison.
- **LLM-only** uses a 150-positive sample per split (450 pairs) with the same negative construction; ranking metrics are not computed for it.

Raw outputs: `results/baselines/`.

## 7. Zero-shot by disease area (S2, seed 0)

| Held-out area | BRIDGE-DR AUROC | BRIDGE-DR AUPRC | R-GCN AUROC | R-GCN AUPRC |
|---|---|---|---|---|
| Neurodegenerative | 0.713 | **0.527** | 0.638 | 0.403 |
| Mental health | 0.735 | **0.583** | 0.656 | 0.418 |
| Cardiovascular | 0.780 | 0.601 | 0.769 | **0.642** |
| Metabolic disorder | 0.711 | **0.526** | 0.666 | 0.459 |
| Autoimmune | 0.811 | 0.642 | 0.859 | **0.747** |
| Cell proliferation | 0.772 | **0.706** | 0.790 | 0.676 |
| Anaemia | 0.695 | 0.466 | 0.727 | **0.599** |
| Adrenal gland | 0.913 | 0.746 | 0.956 | **0.948** |
| Diabetes | 0.590 | **0.474** | 0.565 | 0.370 |
| **Mean ± SD** | | **0.586 ± 0.098** | | 0.585 ± 0.191 |

- **Same mean, steadier:** BRIDGE-DR and R-GCN have the same mean zero-shot AUPRC. BRIDGE-DR wins on 5 of 9 areas and its spread across areas is about half that of R-GCN.
- **Where R-GCN wins:** autoimmune, anaemia, adrenal gland and cardiovascular.

Raw outputs: `results/areas/` (BRIDGE-DR), `results/baselines_areas/rgcn/` (R-GCN).

## 8. Ablations (S1 / S2 neurodegenerative AUPRC)

| Configuration | S1 | S2 |
|---|---|---|
| Full model | 0.908 | 0.527 |
| A1 – no TriCAL | 0.908 | 0.396 |
| A2 – no modality dropout | 0.920 | 0.636 |
| A3 – no FiLM conditioning | 0.911 | 0.492 |
| A6 – random negatives only | 0.782 | 0.371 |

- **Caveat:** the seed check in section 6 shows BRIDGE-DR's S2 AUPRC moving by ±0.15 between seeds. The S2 ablation differences are within that range, so they are not conclusive with 1 seed. The S1 differences are reliable; validation is stable there.
- **TriCAL, FiLM and hard negatives** each help, mostly on zero-shot.
- **A2** scored higher without modality dropout (1 seed; may be noise).
- **A6** falls on S1 because the evaluation includes contraindications as negatives, which a model never trained on them cannot separate.

Raw outputs: `results/ablations/`.


## 9. Explanation checks, cost and candidate targets

**Path fidelity** (`scripts/path_fidelity.py`): the drop in a prediction's score when the edges of its explanation are removed from the graph, averaged over 10 explained predictions that have paths.

| Edges removed | Path fidelity |
|---|---|
| REX paths (attention × gradient) | 0.483 |
| Path-only (same search, no model weights) | 0.101 |
| Random edges at the drug (control) | 0.290 |

REX paths carry much more of the prediction than model-agnostic paths or random edges.

**REX ablations** (`scripts/rex_ablation.py`, 20 predictions, NLI against the retrieved abstracts): claim support 0.183 without verification, 0.020 without retrieval, 0.010 for the LLM alone.

**CPU-only cost** (`scripts/cost_check.py --device cpu`): one training epoch takes about 2 min on CPU (≈ 2 h for 60 epochs), scoring all drugs for one disease 6 s, CFS 0.05 s; peak RAM 3–4 GB.

**Immune / endothelial targets** (`scripts/target_check.py`): a long COVID candidate counts when a PrimeKG target is linked to an immune, inflammatory or endothelial GO process; 5 of the 10 long COVID candidates qualify.

## 10. Literature review

`python scripts/fill_report.py --with-literature` adds a section that reviews knowledge-graph repurposing (Hetionet, PrimeKG, TxGNN, MRDGNN, HAMGNN), KG embedding and GNN link prediction, signature reversal (CMap, L1000), conformal selection and grounded explanations, with 19 references checked against PubMed and arXiv.

## Reproduce

```bash
python scripts/fetch_smiles.py --out data/primekg/drug_smiles.csv
python scripts/run.py --data primekg --smiles-csv data/primekg/drug_smiles.csv --splits s1 s2 --area neurodegenerative \
  --epochs 60 --case-disease "Alzheimer disease" "long COVID-19" --explain-n 5 --llm qwen2.5:3b --out results/primekg
# Re-run evaluation, CFS and REX from saved checkpoints without retraining:
python scripts/run.py ... --resume
```
