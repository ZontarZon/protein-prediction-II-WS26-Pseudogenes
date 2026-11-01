# MMseqs2 setup and usage

MMseqs2 is a command-line tool for sequence searching and clustering. This guide
covers installing it, using it directly, and using the project's WebViewer integration.

## Quick installation

Choose one installation method; compiling is optional. These package commands
follow the [upstream installation guide](https://github.com/soedinglab/MMseqs2#installation):

```bash
# Conda (Linux/macOS)
conda install -c conda-forge -c bioconda mmseqs2

# Homebrew (macOS/Linux)
brew install mmseqs2

# Verify the installed executable
mmseqs version
```

For Conda, activate the environment before starting the WebViewer backend so it
inherits that environment's `PATH`. Alternatively, configure the absolute path
to its `bin/mmseqs` executable in WebViewer.

Precompiled binaries are also available from the upstream guide. Choose a build
for your operating system and CPU: an x86 AVX2 binary requires AVX2 support;
ARM64 needs a compatible ARM64 build. Extract the archive and add its `bin`
directory to `PATH`, or use the executable's absolute path. On Windows, WSL2
with a Linux installation is a practical option; run the backend in WSL too.

MMseqs2 is a separate executable, not a Python package. Installing the project's
Python requirements or checking out `external/MMseqs2` does not install it.

## Compile the bundled source

Run these commands from the repository root. Requirements for the source revision
bundled here (see [CMakeLists.txt](external/MMseqs2/CMakeLists.txt)) are:

- Git, to initialize the submodule.
- CMake **3.15 or newer**, a C/C++ compiler with C++14 support, and a build tool
  such as Make. GCC with OpenMP support is a straightforward Linux choice.
- Rust **1.68.0 or newer** and Cargo, used to build the bundled block aligner.
- OpenMP headers/runtime for the selected compiler; the default build requires it.
- zlib and bzip2 development packages for compressed-file support (optional in
  upstream's build, included in the example below).
- Internet access for any Cargo dependencies not already cached, plus disk space
  and RAM for compilation. CUDA and MPI are not needed for this project's CPU workflow.

Example prerequisites on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake git rustc cargo zlib1g-dev libbz2-dev
cmake --version
rustc --version
cargo --version
```

Check the reported versions: older distribution packages may need updating,
particularly Rust. A current Rust toolchain can be installed through the
[official Rust installer](https://www.rust-lang.org/tools/install).

Initialize and build the source, then install into a persistent user directory:

```bash
git submodule update --init --recursive external/MMseqs2
cmake -S external/MMseqs2 -B /tmp/mmseqs2-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build /tmp/mmseqs2-build --parallel 2
/tmp/mmseqs2-build/bin/mmseqs version
cmake --install /tmp/mmseqs2-build
"$HOME/.local/bin/mmseqs" version
```

Use the absolute path to `~/.local/bin/mmseqs` in WebViewer, or add
`export PATH="$HOME/.local/bin:$PATH"` to your shell startup file and restart the
backend from that environment. The build executable under `/tmp` also works,
but temporary directories may be removed after a reboot. The default build
optimizes for the build machine; use upstream's architecture options if the
binary must run on a different CPU.

The WebViewer Dockerfile installs the Debian MMseqs2 package **inside the backend
container** and verifies `mmseqs version` during the image build. From
`WebViewer/`, run `./docker_run.sh --rebuild` to update an existing image and
replace the saved container. Leave **MMseqs2 executable** set to `mmseqs`.
No host installation or bundled-source compilation is needed for this setup.
See [WebViewer/RUNNING.md](WebViewer/RUNNING.md) for Docker startup details.

## General command-line usage

With `mmseqs` available on `PATH`, these commands run independently of WebViewer:

```bash
# Search query proteins against reference proteins; write tabular alignments.
mmseqs easy-search queries.fasta reference.fasta hits.m8 tmp/search

# Cluster proteins at 50% identity and 80% coverage of both sequences.
mmseqs easy-cluster proteins.fasta clusters tmp/cluster \
  --min-seq-id 0.5 -c 0.8 --cov-mode 0

# Create a reusable reference database and search against it.
mmseqs createdb reference.fasta reference_db
mmseqs easy-search queries.fasta reference_db hits.m8 tmp/search-db
```

`tmp/...` directories hold intermediate files and need sufficient free space.
Clustering produces files including `clusters_cluster.tsv` (representative/member
pairs). These commands search or cluster sequences; the project's learned PE
classification is provided by its separate trainer and classifier. See the
[upstream user guide](https://github.com/soedinglab/MMseqs2/wiki) for command options
and other workflows.

## Using MMseqs2 in this project

**After installing and configuring the executable, starting the MMseqs2 training
run is sufficient.** The project invokes the CLI automatically. You do not need
to run the commands above, prepare MMseqs2 databases manually, or download a
public reference database. MMseqs2 is used specifically by `MMseqs2Trainer` and
`MMseqs2Classifier`; selecting another model does not enable it automatically.

1. Start WebViewer with its normal project dependencies installed.
2. In **New Run**, select **MMseqs2Trainer** and **MMseqs2Classifier**.
3. Select the directory containing your PE-labelled protein FASTA dataset.
4. Leave **MMseqs2 executable** as `mmseqs` if the backend can find it on `PATH`;
   otherwise enter the absolute executable path visible to the backend.
5. Set **MMseqs2 CPU threads** and the desired training/search settings, then start
   the run. Both MMseqs2 and the classifier run on CPU; no GPU is required.

The trainer automatically reads the dataset, clusters sequences, splits whole
clusters into training/validation partitions, builds a reference database from
training entries, searches both partitions against it, and trains a small neural
classifier on the resulting features. For checkpoint inference, it automatically
rebuilds the reference database and searches the supplied sequence.

### Dataset format

Select a dataset directory containing `.fasta`, `.fa`, `.faa`, `.fas` files
(optionally gzip-compressed). Each protein header needs exactly one `PE=1..5`:

```fasta
>protein-a PE=1
MKWVTFISLLFLFSSAYS
>protein-b PE=5
MNNIRRVAILGASV
```

This example illustrates formatting only; training needs both classes in both
cluster partitions. PE 1–4 is class 0; PE 5 is class 1. These are UniProt existence
evidence labels, **not pseudogene or biological authenticity labels**. Duplicate
original IDs are supported through internal sequence IDs. Empty sequences,
non-amino-acid symbols, gaps and stop markers are rejected.

### Features and validation

`easy-cluster` uses configurable identity/coverage thresholds and coverage mode
0. Entire clusters and exact duplicate sequences remain in one partition.
A seeded cluster shuffle selects approximately the requested validation fraction;
large clusters can make the actual percentage differ. If either partition lacks
a class, training fails with an actionable error. Adjust the seed/percentage or
provide more independent clusters. Clustering reduces homology leakage but does
not certify a strict pairwise similarity bound across partitions.

Only training entries form the `createdb` search reference. `easy-search` uses
protein search, exact identity calculation, a configurable E-value/sensitivity,
and at most 1,000 prefilter candidates per query. Features exclude references
with a sequence identical to the query, including duplicate entries. For each
reference label: maximum identity, query coverage, target coverage, log1p bit
score, and log1p retained hit count. The eleventh feature flags no retained hits.
Counts and maxima describe the returned hits, not an exhaustive homology census.
Validation labels never contribute to reference features. A small neural head
learns from these eleven features; no protein language model is downloaded.

### Artifacts and inference

Run outputs contain `mmseqs2_cache/` (databases, FASTA and hits),
`mmseqs2_manifest.json` (input hash, sequence ID mapping, split and parameters),
and `checkpoints/*.pt` plus JSON metric sidecars. Cache keys include inputs,
settings and MMseqs2 version; incomplete command caches are rebuilt. FASTA
records and feature tensors are currently held in memory, so start with a
moderate dataset. MMseqs2 also needs temporary disk space for its databases.

Checkpoints embed classifier weights, optimizer, reference records, feature
version and search/split settings. Resume checks the dataset hash and requires
matching architecture, settings and MMseqs2 version. Stopping before reference
preparation creates a resumable checkpoint that is not yet usable for inference.
Mid-epoch stops resume that epoch using the saved current weights.

For inference, select the checkpoint, enter a single protein sequence/FASTA entry,
and set the executable path if needed. The executable version must match training.
The reference database is rebuilt in temporary storage from the checkpoint;
copying the `.pt` file is sufficient for the model data. Results contain PE-class
probabilities and a no-hit flag. Scores are uncalibrated; no-hit predictions rely
on the learned head's prior. Exported weights alone do not contain the reference
records needed for sequence inference.

## Troubleshooting

- **Executable not found:** run `mmseqs version` in the backend environment, or
  provide an absolute path. Restart the backend after changing its environment.
- **Illegal instruction:** use a binary compatible with the backend CPU, or build
  on that machine.
- **Version mismatch during resume/inference:** use the same MMseqs2 version that
  created the checkpoint; the project checks the reported version string.
- **Missing class in a partition:** supply more independent clusters or adjust
  the split seed/validation percentage.
- **Search or clustering fails:** check the reported command output, available
  RAM, and writable space for run caches and temporary files.

## Verification

```bash
python -m unittest discover -s WebViewer/tests -v
```

Tests use a deterministic CLI double to exercise discovery, training, feature
exclusion, duplicate splitting, portable inference, resume, changed-input
rejection, command failures and stop handling. A real MMseqs2 executable is still
needed to validate search behaviour and performance on biological datasets.
