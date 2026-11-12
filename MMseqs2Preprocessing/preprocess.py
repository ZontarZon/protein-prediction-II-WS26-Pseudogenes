#!/usr/bin/env python3
"""Edit the settings below, then run this file to prepare an MMseqs2 dataset."""

from pathlib import Path
import sys
import tempfile
import torch

# Settings: relative paths are resolved from the directory you run Python in.
INPUT_PATH = Path(
    "../WebViewer/ModelsAndTrainers/Datasets/uniprot_full_fasta/uniprot_sprot.fasta.gz"
)  # FASTA/gzip file or directory
ANNOTATIONS_PATH = Path(
    "../WebViewer/ModelsAndTrainers/Datasets/uniprot_full_fasta/uniprot_pseudogene_caution.tsv"
)
OUTPUT_PATH = Path("./prepared.pt")
CACHE_PATH = None  # None uses OUTPUT_PATH.parent / 'mmseqs2_cache'
EXECUTABLE = "mmseqs"  # Or an absolute path to the native executable
THREADS = 24
# Clustering controls the fixed train/validation split.
IDENTITY = 0.5  # --min-seq-id
CLUSTER_MODE = 0  # --cluster-mode: 0=set cover, 1=connected component, 2=greedy
CLUSTER_SENSITIVITY = 7.5  # -s
CLUSTER_EVALUE = 0.001  # -e
CLUSTER_MAX_SEQS = 1000  # --max-seqs
COVERAGE = 0.8  # -c
COV_MODE = 0  # --cov-mode
VALIDATION_PERCENT = 20
SEED = 7

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "WebViewer"))

from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import (
    DEFAULT_SETTINGS,
    PREPARED_VERSION,
    fingerprint,
    load_records,
    split_records,
    validate_settings,
)
from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Runner

# The steps for preparing the dataset include:
#   1. Set and validate settings
#   2. Set output path
#   3. Fetch annotation scores (UniProt records with the pseudo gene caution
#   and records with evidence at transcription/protein level)
#   4. Set the runner
def prepare_dataset(
    input_path,
    output_path,
    cache,
    executable="mmseqs",
    threads=2,
    settings=None,
    report=print,
    annotation_path=None,
):
    """Select caution-labelled records, cluster them, and save a fixed split."""
    settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    # Ensure required settings are there (identity, coverage, validation
    # percent, seed) and that the included settings values are valid.
    validate_settings(settings)
    # Set output path of prepared dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report("Loading pseudogene-caution and PE1/PE2 records")
    # load_records in MMSeqs2ClassificationData.py. Gets the UniProt Class 0
    # (pseudogene caution) and Class 1 (PE1/PE2 records with evidence at
    # transcript and/or protein level).
    records = load_records(
        input_path, ANNOTATIONS_PATH if annotation_path is None else annotation_path
    )
    # From MMseqs2Runner.py. 
    runner = MMseqs2Runner(executable, threads)
    report("Clustering sequences")
    assignments = runner.clusters(
        records,
        cache,
        settings["identity"],
        settings["coverage"],
        cov_mode=settings.get("cov_mode", 0),
        cluster_mode=settings.get("cluster_mode", 0),
        sensitivity=settings.get("cluster_sensitivity", 7.5),
        evalue=settings.get("cluster_evalue", 0.001),
        max_seqs=settings.get("cluster_max_seqs", 1000),
    )
    train, valid = split_records(records, assignments, settings)
    state = {
        "prepared_version": PREPARED_VERSION,
        "dataset_kind": "mmseqs2_sequence_split",
        "dataset_digest": fingerprint(records),
        "mmseqs_version": runner.version,
        "settings": settings,
        "class_labels": {
            "0": "PE5 + pseudogene caution",
            "1": "PE1/PE2 without caution",
        },
    }
    for name, partition in (("train", train), ("validation", valid)):
        report(f"Saving {name} sequences ({len(partition)} records)")
        state[name] = {
            "targets": torch.tensor([r["target"] for r in partition], dtype=torch.long),
            "records": partition,
        }
    # Publish only complete artifacts, even if an earlier preparation is interrupted.
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    report(f"Saved {output_path}")


def main():
    output = Path(OUTPUT_PATH)
    if output.exists():
        sys.exit(
            "Output already exists; choose a new OUTPUT_PATH to avoid overwriting a prepared dataset"
        )
    settings = {
        "identity": IDENTITY,
        "cluster_mode": CLUSTER_MODE,
        "cluster_sensitivity": CLUSTER_SENSITIVITY,
        "cluster_evalue": CLUSTER_EVALUE,
        "cluster_max_seqs": CLUSTER_MAX_SEQS,  # The data is massive, so we cut back on how much we use.
        "coverage": COVERAGE,
        "cov_mode": COV_MODE,
        "validation_percent": VALIDATION_PERCENT,  # % of the data dedicated to the validation split
        "seed": SEED,  # For replication of the dataset test/val distribution
    }
    cache = (
        Path(CACHE_PATH) if CACHE_PATH is not None else output.parent / "mmseqs2_cache"
    )
    try:
        prepare_dataset(
            INPUT_PATH,
            output,
            cache,
            EXECUTABLE,
            THREADS,
            settings,
            annotation_path=ANNOTATIONS_PATH,
        )
    except (ValueError, OSError, RuntimeError) as error:
        sys.exit(f"Preprocessing failed: {error}")


if __name__ == "__main__":
    main()
