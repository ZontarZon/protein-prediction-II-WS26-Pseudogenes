"""PE-labelled sequences, MMseqs2 cluster splits, and prepared dataset loading."""
import hashlib
import json
import math
import random

import torch

from ModelsAndTrainers.DataLoaders.ProteinClassificationData import ProteinClassificationDataset

DEFAULT_SETTINGS = {'identity': .5, 'coverage': .8, 'cov_mode': 0,
                    'cluster_mode': 0, 'cluster_sensitivity': 7.5,
                    'cluster_evalue': .001, 'cluster_max_seqs': 1000,
                    'validation_percent': 20, 'seed': 7}


def protein_sequence(value):
    lines = str(value).strip().splitlines()
    if lines and lines[0].startswith('>'):
        lines = lines[1:]
    sequence = ''.join(''.join(lines).split()).upper()
    if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYXBZOUJ'):
        raise ValueError('Expected one nonempty protein sequence using amino-acid letters')
    return sequence


def load_records(path, control=None):
    records = []
    for index, record in enumerate(ProteinClassificationDataset(path)):
        if control is not None and not control():
            from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Stopped
            raise MMseqs2Stopped()
        records.append({'key': f's{index}', 'id': record['id'], 'target': record['target'],
                        'sequence': protein_sequence(record['sequence'])})
    if not records:
        raise ValueError('No labelled FASTA records found')
    return records


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def split_records(records, assignments, settings):
    by_key = {r['key']: r for r in records}
    parents = {key: key for key in by_key}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    assigned = set()
    for representative, member in assignments:
        if representative not in by_key or member not in by_key:
            raise ValueError('Unknown sequence ID in MMseqs2 clusters')
        parents[find(member)] = find(representative)
        assigned.add(member)
    if assigned != set(by_key):
        raise ValueError('Incomplete MMseqs2 cluster assignments')
    # Enforce exact-duplicate grouping even if the clustering output changes.
    sequences = {}
    for record in records:
        previous = sequences.setdefault(record['sequence'], record['key'])
        parents[find(record['key'])] = find(previous)
    groups = {}
    for key in by_key:
        groups.setdefault(find(key), []).append(key)
    keys = sorted(groups)
    random.Random(settings['seed']).shuffle(keys)
    if len(keys) < 2:
        raise ValueError('At least two sequence clusters are needed for a holdout')
    desired = max(1, round(len(records) * settings['validation_percent'] / 100))
    validation = set()
    for key in keys[:-1]:
        if len(validation) >= desired:
            break
        validation.update(groups[key])
    train = [r for r in records if r['key'] not in validation]
    valid = [r for r in records if r['key'] in validation]
    if any({r['target'] for r in split} != {0, 1} for split in (train, valid)):
        raise ValueError('Both PE classes are required in each cluster split. Add independent clusters, or adjust the split seed/percentage.')
    return train, valid


PREPARED_VERSION = 3


def validate_settings(settings):
    required = {'identity', 'coverage', 'validation_percent', 'seed'}
    if not required <= set(settings) or set(settings) - set(DEFAULT_SETTINGS):
        raise ValueError('Incomplete preprocessing settings')
    if any(type(settings[k]) not in (int, float) or not math.isfinite(settings[k])
           for k in settings):
        raise ValueError('Preprocessing settings must be finite numbers')
    if not (0 <= settings['identity'] <= 1 and 0 < settings['coverage'] <= 1
            and type(settings['validation_percent']) is int and 1 <= settings['validation_percent'] <= 50
            and type(settings['seed']) is int and settings['seed'] >= 0
            and type(settings.get('cov_mode', 0)) is int and 0 <= settings.get('cov_mode', 0) <= 5):
        raise ValueError('Invalid preprocessing thresholds or split settings')
    if (type(settings.get('cluster_mode', 0)) is not int or settings.get('cluster_mode', 0) not in (0, 1, 2)
            or not 1 <= settings.get('cluster_sensitivity', 7.5) <= 7.5
            or settings.get('cluster_evalue', .001) <= 0
            or type(settings.get('cluster_max_seqs', 1000)) is not int
            or settings.get('cluster_max_seqs', 1000) <= 0):
        raise ValueError('Invalid clustering settings')


def load_prepared(path):
    """Load an artifact, preferring prepared.pt or a single .pt file in a directory."""
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        canonical = path / 'prepared.pt'
        if canonical.is_file():
            path = canonical
        else:
            candidates = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.pt')
            if len(candidates) > 1:
                names = ', '.join(p.name for p in candidates)
                raise ValueError(f'Multiple .pt files in {path}: {names}. Name the intended dataset prepared.pt or select its file path directly.')
            if not candidates:
                raise ValueError(f'No preprocessed .pt dataset found in {path}. Run MMseqs2Preprocessing/preprocess.py first.')
            path = candidates[0]
    if not path.is_file():
        raise ValueError(f'Preprocessed dataset not found: {path}. Run MMseqs2Preprocessing/preprocess.py first.')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or state.get('prepared_version') != PREPARED_VERSION or state.get('dataset_kind') != 'mmseqs2_sequence_split':
        raise ValueError('This trainer requires a sequence-split dataset. Regenerate with MMseqs2Preprocessing/preprocess.py; legacy 11-feature datasets lack validation sequences.')
    try:
        validate_settings(state['settings'])
        if not isinstance(state['mmseqs_version'], str) or not isinstance(state['dataset_digest'], str):
            raise ValueError('Missing preprocessing metadata')
        split_keys, split_sequences = [], []
        for name in ('train', 'validation'):
            partition = state[name]
            targets, records = partition['targets'], partition['records']
            if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.long
                    or targets.ndim != 1 or set(targets.tolist()) != {0, 1}
                    or len(records) != len(targets)):
                raise ValueError(f'Invalid {name} labels or records')
            keys, sequences = set(), set()
            for record, target in zip(records, targets.tolist()):
                if record['key'] in keys or record['target'] != target:
                    raise ValueError('Duplicate IDs or inconsistent labels')
                if protein_sequence(record['sequence']) != record['sequence']:
                    raise ValueError('Invalid protein sequence')
                keys.add(record['key'])
                sequences.add(record['sequence'])
            split_keys.append(keys)
            split_sequences.append(sequences)
        if split_keys[0] & split_keys[1] or split_sequences[0] & split_sequences[1]:
            raise ValueError('Overlapping train/validation records or sequences')
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError('Malformed preprocessed MMseqs2 dataset') from error
    return state, digest


PREVIEW_EXTENSIONS = ('.pt',)


def preview_records(path):
    """Show sequence rows from the fixed MMseqs2 split."""
    state, _ = load_prepared(path)
    for split in ('train', 'validation'):
        for record in state[split]['records']:
            yield {'split': split, 'id': record['id'], 'key': record['key'],
                   'target': record['target'],
                   'label': 'PE 1–2' if record['target'] == 0 else 'PE 5',
                   'sequence': record['sequence'], 'length': len(record['sequence'])}
