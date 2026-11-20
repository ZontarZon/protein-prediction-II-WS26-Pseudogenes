# Prepare MMseqs2 sequence splits for pLM training

## Clustering files

- `data.fasta` contains the original query protein sequences.
- `blast_results/*.xml` contains one BLAST result per query.
- `xmlProcess.py` pairs each query with its highest-scoring BLAST hit after excluding accessions in `data.fasta` and hits annotated PE 3, 4, or 5. Queries without an eligible hit are skipped.
- `paired_hits.fasta` contains those query and hit sequences and serves as the clustering input.
- `mmseqs2.py` builds the pairs, runs MMseqs2, and saves the JSON and FASTA exports.
- `outputs/clusters.json` lists clusters using the representative’s 1-based input position as `cluster_id`, with each item’s sequence, header, parsed UniProt fields, and 1,024-value ProtT5 embedding. The `tag` is `pseudogene` when the accession appears in `data.fasta` and `regular` otherwise.
- `outputs/cluster_representatives.fasta` contains one representative sequence per cluster, labeled by input record number.
- `outputs/cluster_sequences.fasta` contains all member sequences grouped by cluster, labeled by input record number.

Run clustering and embedding with `MMseqs2Preprocessing/.venv/bin/python MMseqs2Preprocessing/mmseqs2.py` after installing the requirements below into the virtual environment. The first run downloads the ProtT5 checkpoint.

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

## Local sequential BLASTP searches

Install NCBI BLAST+ and the Python dependencies on your machine:

```bash
sudo apt install ncbi-blast+
pip install -r MMseqs2Preprocessing/requirements-blast.txt
python MMseqs2Preprocessing/blastJob.py
```

Edit the settings at the top of `blastJob.py` for query/output paths, database
location, threads, E-value (default 10), and maximum hits (default 500).
Biopython reads the query FASTA; the native BLAST+ executables run the searches.
No Docker or remote API is used.

The default database source is
`/mnt/Daten_Programme/Bibliotheken/Downloads/UniProtKB.gz`, a gzip-compressed
UniProt JSON export with a `results` array. On the first run, `ijson` streams
one entry at a time directly into `makeblastdb`, without loading the archive into
memory or saving an uncompressed FASTA. This reads the entire archive once and
can take hours. Allow substantial additional disk space for the BLAST index;
the script reports free space and asks before building in an interactive terminal.
Native tool messages go to `outputs/blast.log`, and Python progress appears both
there and in the terminal (including every 100,000 database proteins).

The completed index lives in `local_blast_db/` beside the script and is reused.
A completion marker records the source path, size and modification time. If the
source changes, select a new `DATABASE_DIR`. An interrupted build has no marker
and is rebuilt on the next run. Do not modify the source or share the same database
or output directory between concurrent runs.

Queries run sequentially, each using the configured CPU threads. Output stays
separated by input position and sanitized ID, including duplicate IDs:

```text
MMseqs2Preprocessing/outputs/
  blast.log
  0001_sp_P12345_PROTEIN/
    sequence.fasta
    result.json
```

Each FASTA preserves the original query description. `result.json` contains
native BLAST single-file JSON (`-outfmt 15`), including alignments and hit details;
its internal schema differs from the old EBI service JSON. JSON is validated
before a temporary file replaces the result. No-hit searches still produce JSON.
Invalid queries are logged and skipped; native tool errors stop the run.
Ctrl+C stops the process and keeps completed results. Rerunning searches every
query again and replaces matching results; use a new `OUTPUT_DIR` to retain a run.
