# Prepare an MMseqs2 training dataset

Run from the repository root with Python 3.11 or newer. Docker and the WebViewer
backend dependencies are not needed. Install the native `mmseqs` executable using
[the installation guide](../MMseqs2SetupAndUsage.md); pip does not install it.

```bash
python3 -m venv MMseqs2Preprocessing/.venv
source MMseqs2Preprocessing/.venv/bin/activate
pip install -r MMseqs2Preprocessing/requirements.txt
python MMseqs2Preprocessing/preprocess.py
```

Before running, edit the settings at the top of `preprocess.py`, especially
`INPUT_PATH` and `OUTPUT_PATH`. No command-line arguments are needed.

`INPUT_PATH` accepts a FASTA file or a directory of FASTA files, including gzip.
Headers must contain exactly one `PE=1..5` label. PE 1–2 maps to class 0, PE 5 to
class 1. PE 3 and 4 entries are skipped; these are evidence labels, not pseudogene labels. Both resulting
partitions must contain both classes.

Set `EXECUTABLE` to an absolute path when `mmseqs` is not on PATH. All settings
are variables at the top of the script: `THREADS`, `IDENTITY`, `COVERAGE`,
`SENSITIVITY`, `EVALUE`, `MAX_SEQS`, `COV_MODE`, `VALIDATION_PERCENT`, and `SEED`. Defaults are 2 threads,
0.5 identity, 0.8 coverage, 7.5 sensitivity, 0.001 E-value, 20% validation,
and seed 7. Set `CACHE_PATH` to choose the reusable working directory;
`None` uses `mmseqs2_cache` beside the output. Relative paths are resolved from
the directory you run Python in. Allow enough RAM for all records/features and
disk space for MMseqs2 temporary databases. The output parent is created
automatically; existing output files are rejected.

Preparation clusters sequences, assigns entire clusters (and exact duplicates)
to one split, and searches both splits against only the training reference.
It excludes sequence-identical hits and saves the eleven existing homology
features. Clustering reduces leakage but does not guarantee a strict pairwise
similarity bound between splits.

The single portable `.pt` file includes feature tensors, labels, split IDs,
training reference sequences, input digest, settings, and MMseqs2 version.
Keep this file; the working cache is optional after preparation completes.

In WebViewer, make the output directory available under a configured dataset
root (and mounted into the backend if using Docker). Select that dataset with
**MMseqs2Trainer** and **MMseqs2Classifier**. The trainer prefers `prepared.pt` in
the selected directory. If absent, it accepts a single `.pt` file with any name
(such as `preProcessed.pt`). Multiple `.pt` files require naming the intended
dataset `prepared.pt` or selecting a direct file path programmatically.
Training uses CPU PyTorch only and does not invoke MMseqs2. Change epochs,
batch size, learning rate, shuffle seed, or head size without preprocessing again.
The shuffle seed controls training order; the preprocessing seed controls the
fixed cluster split.

Prepare a new file when inputs/labels, reference membership, split, feature
implementation, or MMseqs2 settings/version change. Each cross-validation fold
needs its own training reference and features. Resume verifies the prepared
file digest; keep the original file when resuming. Legacy checkpoints without
that digest are checked against input digest, settings, and MMseqs2 version.

Checkpoints embed reference records so inference remains portable. Predicting
new sequences still requires the same MMseqs2 version used for preprocessing:
it searches the new query against the checkpoint's training reference. MMseqs2
is not a reader for the saved tensor file; PyTorch loads that file directly.

Run the focused integration checks with:

```bash
python -m unittest discover -s MMseqs2Preprocessing/tests -v
```

Prepared files from the previous PE 1–4 policy are rejected. Regenerate them
with this script to use only PE 1–2 versus PE 5.

Search uses `-s 7.5 -e 0.001 --max-seqs 1000 -c 0.8 --cov-mode 0`
by default. Coverage and coverage mode also apply to clustering. These settings
are saved with the dataset and reused for inference. Older prepared files keep
their original search behaviour (no search coverage filter). Regenerate the
dataset to apply the new search thresholds.

Clustering also has separate top-level variables: `CLUSTER_MODE`,
`CLUSTER_SENSITIVITY`, `CLUSTER_EVALUE`, and `CLUSTER_MAX_SEQS`. Search uses
`SENSITIVITY`, `EVALUE`, and `MAX_SEQS`; `COVERAGE` and `COV_MODE` apply to both.
Changing clustering settings changes the split; changing search settings changes
the feature tensors. All these settings are recorded in the prepared artifact
and included in the relevant cache keys.

Protein database/search types, exact-identity alignment mode, and the output
column format remain fixed because the PE feature loader requires that contract.
Classifier epochs, learning rate, batch size, and shuffle seed remain training
settings in WebViewer.
