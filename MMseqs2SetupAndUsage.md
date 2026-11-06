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

Run the standalone [preprocessing tool](MMseqs2Preprocessing/README.md) once to
cluster PE-labelled FASTA sequences and save sequences, labels, fixed split
membership, and provenance in a portable `.pt` file.

Select the prepared directory with **MMseqs2Trainer** and **MMseqs2Classifier**.
The trainer preserves the cluster split, embeds sequences using frozen ProtT5,
and trains a classification head on those embeddings. MMseqs2 is required only
for preprocessing. Inference uses ProtT5 directly. See the tool README for setup,
settings, and the required regeneration of legacy eleven-feature datasets.
