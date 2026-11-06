#!/usr/bin/env python3
"""Edit the settings below, then run this file to prepare an MMseqs2 dataset."""
from pathlib import Path
import sys

# Settings: relative paths are resolved from the directory you run Python in.
INPUT_PATH = Path('../WebViewer/ModelsAndTrainers/Datasets/uniprot_full_fasta/uniprot_sprot.fasta.gz')  # FASTA/gzip file or directory
OUTPUT_PATH = Path('./prepared.pt')
CACHE_PATH = None  # None uses OUTPUT_PATH.parent / 'mmseqs2_cache'
EXECUTABLE = 'mmseqs'  # Or an absolute path to the native executable
THREADS = 24
IDENTITY = 0.5
COVERAGE = 0.8
SENSITIVITY = 5.7
EVALUE = 0.001
VALIDATION_PERCENT = 20
SEED = 7

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'WebViewer'))

from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import prepare_dataset


def main():
    output = Path(OUTPUT_PATH)
    if output.exists():
        sys.exit('Output already exists; choose a new OUTPUT_PATH to avoid overwriting a prepared dataset')
    settings = {
        'identity': IDENTITY,
        'coverage': COVERAGE,
        'sensitivity': SENSITIVITY,
        'evalue': EVALUE,
        'validation_percent': VALIDATION_PERCENT,
        'seed': SEED,
    }
    cache = Path(CACHE_PATH) if CACHE_PATH is not None else output.parent / 'mmseqs2_cache'
    try:
        prepare_dataset(INPUT_PATH, output, cache, EXECUTABLE, THREADS, settings)
    except (ValueError, OSError, RuntimeError) as error:
        sys.exit(f'Preprocessing failed: {error}')


if __name__ == '__main__':
    main()
