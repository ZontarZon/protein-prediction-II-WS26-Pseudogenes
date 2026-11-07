# MLP head experiments (ProtT5 PE classifier)

Search for a classification head on frozen ProtT5 per-protein embeddings, evaluated on a
class-balanced Swiss-Prot subset (all PE 5 vs. an equal random sample of PE 1–2) with
homology-grouped 5-fold cross-validation. Results and the Chinese report live in
`docs/mlp_experiments/` (`results.jsonl`, `report.html`, PNG charts).

Data products (gitignored) are written to `WebViewer/ModelsAndTrainers/Datasets/mlp_balanced/`:
`subset.fasta`, `subset.tsv` (accession, pe, label, length), `seed.txt`, `clusters_cluster.tsv`,
`embeddings.npz`.

## Commands

Run from this directory with the project environment:

```bash
PY=/root/miniforge3/envs/pp2-pseudo/bin/python
cd scripts/mlp_experiments
mkdir -p logs   # campaign logs (not committed); wrap long steps as `nohup $PY ... > logs/<step>.log 2>&1 &`
$PY prepare_dataset.py                       # balanced subset (seed 20261106) + mmseqs easy-cluster 0.3/0.8
HF_HUB_CACHE=~/.cache/huggingface/hub $PY embed.py   # ProtT5 embeddings via ProteinPEClassifier.embed (~2 min on GPU)
$PY search.py baselines                      # (a) repo defaults, (b) repo head + early stopping, (c) logistic regression, (b+) tuned
$PY search.py random --n 300 --rng 1         # broad random search
$PY search.py refine --n 150 --top 10 --rng 2
$PY search.py widen --top 6                  # residual, seed ensembles, SWA, cosine, mixup, refit, ...
$PY search.py custom --configs configs/seeds.json    # baselines + selected configs on seeds 1/2, fixed-epoch variants
$PY search.py custom --configs configs/widen2.json   # RBF-SVM reference, 10-head ensemble, epoch/width/label-smoothing checks
$PY make_report.py                           # regenerate docs/mlp_experiments/report.html + charts
```

`search.py seeds --top K` is the generic alternative to the custom file: it re-runs every
baseline and the K best seed-0 search configs on seeds 1 and 2. A seed `s` changes both the
CV split (`StratifiedGroupKFold(random_state=s)`) and model initialisation/shuffling; all
configs share the split for a given seed, so fold-wise paired comparisons are valid.

## Protocol

* Outer: `StratifiedGroupKFold(5)` over MMseqs2 clusters (min-seq-id 0.3, coverage 0.8).
* Inner: 1/8 of each outer training fold (by cluster) is held out for early stopping on AUROC
  (and for picking C for logistic regression). The outer test fold is never used for selection.
* `refit=True`: after early stopping, retrain on the whole outer training fold for the selected
  number of epochs. `ensemble=k`: average the probabilities of k independently seeded heads.
* Metrics per fold: AUROC, AUPRC, accuracy, balanced accuracy, MCC, F1 (threshold 0.5, positive = PE 5).

`configs/seeds.json` and `configs/widen2.json` were written during the campaign from the seed-0
results (they embed the exact configs, so they replay without re-running the selection).
`docs/mlp_experiments/chosen.json` names the configuration deployed to `ProteinPEClassifier`;
`docs/mlp_experiments/conclusion_zh.html` holds the hand-written conclusion inserted into the report.

## Outcome

Saturated: the best single head (2×512, LayerNorm, GELU, dropout 0.5, AdamW lr 1e-4, bs 32,
6 epochs) beats baseline (b) by +0.52 AUROC points over 15 paired folds (13/15 wins); a 10-head
ensemble reaches +0.61. Logistic regression and an RBF-SVM land in the same 0.951–0.957 band.

## Phase 2: representation experiments

```bash
$PY embed_residues.py              # per-residue ProtT5 (same chunking/cleaning as embed) -> residues.h5, fp16, ~2.3 GB
$PY repr_search.py features        # pooled feature cache (mean/max/std/len/first16/last16/seg4/seg8)
$PY repr_search.py pooled --workers 6
$PY repr_search.py la --workers 2  # light attention; ~4 GB GPU per worker (all residues are GPU-resident)
$PY repr_search.py tf --workers 2 --seeds 0          # transformer screen on seed 0 ...
$PY repr_search.py tf --workers 2 --seeds 1,2 --only <best two names>   # ... then the two best on seeds 1/2
$PY oof_predictions.py             # out-of-fold predictions of phase-1 reference configs
```

`run_phase2_queue.sh` runs pooled -> transformer screen -> transformer confirmation one after another.
Never run two GPU jobs at the same time: the light-attention and transformer workers each keep
all residues on the GPU. Out-of-fold P(PE5) for every named run is saved to
`Datasets/mlp_balanced/predictions/<name>_s<seed>.npy`; `pseudogene_caution_pe5.txt` (UniProt REST,
PE 5 with the "Could be the product of a pseudogene" caution) drives the label-quality table.
