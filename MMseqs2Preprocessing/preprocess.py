#!/usr/bin/env python3
"""Edit the settings below, then run this file to prepare an MMseqs2 dataset."""
from pathlib import Path
import sys
import tempfile

import torch

# Settings: relative paths are resolved from the directory you run Python in.
INPUT_PATH = Path('../WebViewer/ModelsAndTrainers/Datasets/uniprot_full_fasta/uniprot_sprot.fasta.gz')  # FASTA/gzip file or directory
OUTPUT_PATH = Path('./prepared.pt')
CACHE_PATH = None  # None uses OUTPUT_PATH.parent / 'mmseqs2_cache'
EXECUTABLE = 'mmseqs'  # Or an absolute path to the native executable
THREADS = 24
# Clustering controls the fixed train/validation split.
IDENTITY = 0.5  # --min-seq-id
CLUSTER_MODE = 0  # --cluster-mode: 0=set cover, 1=connected component, 2=greedy
CLUSTER_SENSITIVITY = 7.5  # -s
CLUSTER_EVALUE = 0.001  # -e
CLUSTER_MAX_SEQS = 1000  # --max-seqs
# Search controls feature generation; saved values are reused for inference.
COVERAGE = 0.8  # -c (clustering and search)
COV_MODE = 0  # --cov-mode
MAX_SEQS = 1000  # --max-seqs (search)
SENSITIVITY = 7.5  # -s
EVALUE = 0.001
VALIDATION_PERCENT = 20
SEED = 7

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'WebViewer'))

from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import (
    DEFAULT_SETTINGS, FEATURE_VERSION, PREPARED_VERSION,
    features, fingerprint, load_records, split_records, validate_settings,
)
from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Runner


def prepare_dataset(input_path, output_path, cache, executable='mmseqs', threads=2, settings=None, report=print):
    """Cluster labelled FASTA records and save portable train/validation features."""
    settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    validate_settings(settings)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report('Loading PE-labelled FASTA records')
    records = load_records(input_path)
    runner = MMseqs2Runner(executable, threads)
    report('Clustering sequences')
    assignments = runner.clusters(records, cache, settings['identity'], settings['coverage'], cov_mode=settings.get('cov_mode', 0),
                                  cluster_mode=settings.get('cluster_mode', 0),
                                  sensitivity=settings.get('cluster_sensitivity', 7.5),
                                  evalue=settings.get('cluster_evalue', .001),
                                  max_seqs=settings.get('cluster_max_seqs', 1000))
    train, valid = split_records(records, assignments, settings)
    state = {'prepared_version': PREPARED_VERSION, 'feature_version': FEATURE_VERSION,
             'dataset_digest': fingerprint(records), 'mmseqs_version': runner.version,
             'settings': settings, 'reference': train}
    for name, partition in (('train', train), ('validation', valid)):
        report(f'Preparing {name} features ({len(partition)} sequences)')
        hits = runner.search(partition, train, cache, settings['sensitivity'], settings['evalue'],
                             max_seqs=settings.get('max_seqs', 1000),
                             coverage=settings['coverage'] if 'cov_mode' in settings else 0.,
                             cov_mode=settings.get('cov_mode', 0))
        state[name] = {'features': features(partition, train, hits),
                       'targets': torch.tensor([r['target'] for r in partition], dtype=torch.long),
                       'records': [{'key': r['key'], 'id': r['id']} for r in partition]}
    # Publish only complete artifacts, even if an earlier preparation is interrupted.
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    report(f'Saved {output_path}')


def main():
    output = Path(OUTPUT_PATH)
    if output.exists():
        sys.exit('Output already exists; choose a new OUTPUT_PATH to avoid overwriting a prepared dataset')
    settings = {
        'identity': IDENTITY,
        'cluster_mode': CLUSTER_MODE,
        'cluster_sensitivity': CLUSTER_SENSITIVITY,
        'cluster_evalue': CLUSTER_EVALUE,
        'cluster_max_seqs': CLUSTER_MAX_SEQS,
        'coverage': COVERAGE,
        'cov_mode': COV_MODE,
        'max_seqs': MAX_SEQS,
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
