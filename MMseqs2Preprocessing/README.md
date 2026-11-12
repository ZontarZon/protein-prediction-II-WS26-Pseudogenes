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

Export the UniProt search results for
`cc_caution:"Could be the product of a pseudogene"` as a TSV,
including the accession (`Entry`) and caution (`Caution` or `Annotation (caution)`)
columns. Use the same Swiss-Prot release as the FASTA. The default input is
already Swiss-Prot, so a `reviewed:true` filter is redundant; use it only if
querying all UniProtKB entries. The FASTA header does not contain caution
comments. Set `INPUT_PATH`,
`ANNOTATIONS_PATH`, and `OUTPUT_PATH` at the top of `preprocess.py`, then run:

```bash
python MMseqs2Preprocessing/preprocess.py
```

Relative paths are resolved from your working directory. `INPUT_PATH` accepts
plain/gzip FASTA or a directory of FASTA files. Preprocessing matches FASTA
accessions to the TSV. Only PE5 entries carrying “Could be the product of a
pseudogene” map to class 0. PE1/PE2 entries absent from the caution TSV map to
class 1; PE1/PE2 entries that do carry the caution are excluded. Other entries
are skipped. The TSV must list caution-tagged entries across all PE levels so
those PE1/PE2 entries can be excluded. This does not yet select the closest
PE1/PE2 proteins.

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

## Train without WebViewer

Install the training dependencies and run from the repository root:

```bash
pip install -r models/requirements.txt
python -m models.ModelsAndTrainers.Trainers.PLM_MLPTrainer --training-data path/to/prepared.pt
```

Edit the normal `hparams` dictionary at the top of
`models/ModelsAndTrainers/Trainers/PLM_MLPTrainer.py` to set epochs, batch size,
learning rate, weight decay, seed, hidden sizes, dropout, normalization, and
activation, as well as input/output paths and device. Defaults use batch size 32
and CUDA when available, otherwise CPU. `--device cpu` and `--run-dir path/to/run`
override the device and output directory. A dataset directory is also accepted:
the loader prefers `prepared.pt`, otherwise requires a single `.pt` file.

Training preserves the prepared split and requires no MMseqs2 executable.
It uses `Rostlab/prot_t5_xl_half_uniref50-enc`: a frozen encoder, mean pooling
over all residues using 1024-residue chunks, and a trainable MLP head. The encoder
stays in evaluation mode while the head trains. Model weights download on first
use; they must be cached for offline training. Loss and accuracy print after each
training/validation pass, and checkpoints go into `run_dir/checkpoints`.

Resume with the same architecture and shuffle seed in `hparams`; set `epochs`
to the desired total number of epochs:

```bash
python -m models.ModelsAndTrainers.Trainers.PLM_MLPTrainer --training-data path/to/prepared.pt --resume runs/plm_mlp/checkpoints/checkpoint_0001.pt
```

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
python -m unittest discover -s models/tests -v
```

## Sequential BLASTP searches

Use Python 3.12+ and install `MMseqs2Preprocessing/requirements.txt` in your
virtual environment. Edit the settings at the top of `blastJob.py`: set `EMAIL`
to your real contact address and choose input/output paths, batch size, polling
interval, submission delay, and request timeout. Then run without arguments:

```bash
python MMseqs2Preprocessing/blastJob.py
```

The script reads the FASTA in small batches, submitting one protein at a time.
It waits 30 seconds before the first status check and between subsequent checks,
downloads the completed XML directly into `OUTPUT_DIR`, then pauses 10 seconds
before continuing. These delays are configurable. Filenames include input position
and FASTA ID. `blast.log` records submitted job IDs and short failure messages
identifying the affected record. Invalid sequences and terminal job failures are
logged and processing continues. Network/API request errors stop the run because
a remote job could still be running. Ctrl+C stops processing.

Rate-limited status checks retry up to `STATUS_RETRIES` times, with pauses
starting at `RETRY_DELAY` seconds and doubling. This includes EBI scheduler
rate limits wrapped in HTTP 400 responses. Error logs include the response body.
Submission requests are never retried automatically.

This simplified script has no resume tracking. Running it
again submits the input again and replaces matching result filenames; use a new
output directory when keeping separate runs.

Offline tests (no API submissions):

```bash
python -m unittest discover -s MMseqs2Preprocessing/tests -v
```
