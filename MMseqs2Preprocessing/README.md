# Prepare MMseqs2 sequence splits for pLM training

MMseqs2 clusters sequences to create a fixed, similarity-aware training/validation
split. The prepared `.pt` file stores **sequences, labels, and split membership**.
Training runs those sequences through frozen ProtT5 and trains a classification
head on its 1024-dimensional embeddings. No eleven-feature homology input or
reference search is used by this pipeline.

Use Python 3.11+ and install the native `mmseqs` executable using the
[installation guide](../MMseqs2SetupAndUsage.md). Docker and the backend Python
dependencies are not needed for preprocessing.

```bash
python3 -m venv MMseqs2Preprocessing/.venv
source MMseqs2Preprocessing/.venv/bin/activate
pip install -r MMseqs2Preprocessing/requirements.txt
```

Edit the variables at the top of `preprocess.py`, particularly `INPUT_PATH` and
`OUTPUT_PATH`, then run:

```bash
python MMseqs2Preprocessing/preprocess.py
```

Relative paths are resolved from your working directory. `INPUT_PATH` accepts
plain/gzip FASTA or a directory of FASTA files. Headers must contain exactly one
`PE=1..5` field. PE 1–2 maps to class 0; PE 5 to class 1; PE 3–4 are skipped.
These are existence-evidence classes, not biological authenticity labels.

The settings block controls the executable, threads, identity, coverage,
coverage mode, cluster mode, clustering sensitivity/E-value/max-seqs, validation
percentage, and split seed. `CACHE_PATH = None` uses `mmseqs2_cache` beside the
output. Cache keys account for clustering parameters, inputs, and MMseqs2 version.
Existing output files are rejected; choose a new filename or move the old file.

Whole clusters and exact duplicate sequences stay in one partition. Both classes
must appear in each partition. If they do not, adjust the split seed/percentage
or provide more independent clusters. Cluster grouping reduces homology leakage;
it does not certify a strict pairwise similarity bound between partitions.
Records remain in memory; allow enough RAM and disk space for MMseqs2 clustering.

Put the output directory under a configured WebViewer dataset root, mounted into
the backend if using Docker. Select **MMseqs2Trainer** and **MMseqs2Classifier**.
The loader prefers `prepared.pt`, otherwise accepts a single `.pt` file in the
selected directory. The data browser displays sequences, labels, and splits.

Training preserves the prepared split and requires no MMseqs2 executable.
It uses `Rostlab/prot_t5_xl_half_uniref50-enc` with the existing ProteinPEClassifier
embedding implementation: frozen encoder, mean pooling over all residues using
1024-residue chunks, and a trainable head. Select CPU or CUDA in WebViewer.
The backend needs its normal ProtT5 dependencies and access to the model weights
(downloaded on first use). Training batch size defaults to 2; increase it to fit
your hardware. The encoder stays in evaluation mode even while the head trains.

Checkpoints save the head, encoder identifier, optimizer, random state, dataset
digest, and preprocessing provenance. Resume checks the original prepared file
and settings. Inference embeds the supplied sequence with ProtT5 and applies the
head; no MMseqs2 reference database is required. Encoder weights come from the
same pretrained model rather than being duplicated in every checkpoint.

**Regenerate legacy eleven-feature `.pt` datasets.** They omit validation
sequences and cannot be converted safely into this format on their own. Legacy
homology-head checkpoints also require a new training run. The loader rejects
them with an actionable message.

```bash
python -m unittest discover -s MMseqs2Preprocessing/tests -v
```
