"""Independent PE-labelled FASTA and homology feature preparation."""
import hashlib
import json
import math
import random

import torch

from ModelsAndTrainers.DataLoaders.ProteinClassificationData import ProteinClassificationDataset

FEATURE_VERSION = 1
FEATURE_COUNT = 11
DEFAULT_SETTINGS = {'identity': .5, 'coverage': .8, 'sensitivity': 5.7, 'evalue': .001,
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


def features(queries, reference, hit_path):
    """Five maxima/count features per label, plus an explicit no-hit flag.

    Identity/qcov/tcov are fractions; bits and retained hit counts use log1p.
    Exclude all sequence-identical references for train/inference consistency.
    """
    query_map = {r['key']: r for r in queries}
    reference_map = {r['key']: r for r in reference}
    values = {key: [0.] * FEATURE_COUNT for key in query_map}
    seen = set()
    with hit_path.open() as stream:
        for line in stream:
            query, target, *numbers = line.rstrip('\n').split('\t')
            if query not in query_map or target not in reference_map or len(numbers) != 4:
                raise ValueError('Invalid MMseqs2 search result')
            if (query, target) in seen:
                continue
            seen.add((query, target))
            if query_map[query]['sequence'] == reference_map[target]['sequence']:
                continue
            identity, qcov, tcov, bits = map(float, numbers)
            if not all(math.isfinite(x) for x in (identity, qcov, tcov, bits)) or not all(0 <= x <= 1 for x in (identity, qcov, tcov)) or bits < 0:
                raise ValueError('Invalid MMseqs2 feature values')
            offset = reference_map[target]['target'] * 5
            row = values[query]
            for i, value in enumerate((identity, qcov, tcov, math.log1p(bits))):
                row[offset + i] = max(row[offset + i], value)
            row[offset + 4] += 1
    for row in values.values():
        row[10] = float(row[4] + row[9] == 0)
        row[4], row[9] = math.log1p(row[4]), math.log1p(row[9])
    return torch.tensor([values[r['key']] for r in queries], dtype=torch.float32)


PREPARED_VERSION = 2


def validate_settings(settings):
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError('Incomplete preprocessing settings')
    if any(type(settings[k]) not in (int, float) or not math.isfinite(settings[k])
           for k in settings):
        raise ValueError('Preprocessing settings must be finite numbers')
    if not (0 <= settings['identity'] <= 1 and 0 < settings['coverage'] <= 1
            and 1 <= settings['sensitivity'] <= 7.5 and settings['evalue'] > 0
            and type(settings['validation_percent']) is int and 1 <= settings['validation_percent'] <= 50
            and type(settings['seed']) is int and settings['seed'] >= 0):
        raise ValueError('Invalid preprocessing thresholds or split settings')


def prepare_dataset(input_path, output_path, cache, executable='mmseqs', threads=2, settings=None, report=print):
    """Prepare portable training features; MMseqs2 is only invoked here/in inference."""
    from pathlib import Path
    import tempfile
    from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Runner

    settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    validate_settings(settings)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report('Loading PE-labelled FASTA records')
    records = load_records(input_path)
    runner = MMseqs2Runner(executable, threads)
    report('Clustering sequences')
    assignments = runner.clusters(records, cache, settings['identity'], settings['coverage'])
    train, valid = split_records(records, assignments, settings)
    state = {'prepared_version': PREPARED_VERSION, 'feature_version': FEATURE_VERSION,
             'dataset_digest': fingerprint(records), 'mmseqs_version': runner.version,
             'settings': settings, 'reference': train}
    for name, partition in (('train', train), ('validation', valid)):
        report(f'Preparing {name} features ({len(partition)} sequences)')
        hits = runner.search(partition, train, cache, settings['sensitivity'], settings['evalue'])
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


def load_prepared(path):
    """Load a validated artifact (or dataset directory containing prepared.pt)."""
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        path = path / 'prepared.pt'
    if not path.is_file():
        raise ValueError(f'Preprocessed dataset not found: {path}. Run MMseqs2Preprocessing/preprocess.py first.')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or state.get('prepared_version') != PREPARED_VERSION or state.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported preprocessed MMseqs2 dataset version. Regenerate with MMseqs2Preprocessing/preprocess.py (PE 1–2 versus PE 5).')
    try:
        validate_settings(state['settings'])
        if not state['reference'] or not isinstance(state['mmseqs_version'], str) or not isinstance(state['dataset_digest'], str):
            raise ValueError('Missing preprocessing metadata or reference')
        reference_keys = set()
        for record in state['reference']:
            if record['target'] not in (0, 1) or protein_sequence(record['sequence']) != record['sequence']:
                raise ValueError('Invalid reference record')
            reference_keys.add(record['key'])
        split_keys = []
        for name in ('train', 'validation'):
            partition = state[name]
            values, targets = partition['features'], partition['targets']
            if (not isinstance(values, torch.Tensor) or values.dtype != torch.float32
                    or values.ndim != 2 or values.shape != (len(targets), FEATURE_COUNT)
                    or not torch.isfinite(values).all() or (values < 0).any()
                    or not isinstance(targets, torch.Tensor) or targets.dtype != torch.long
                    or targets.ndim != 1 or set(targets.tolist()) != {0, 1}
                    or len(partition['records']) != len(targets)):
                raise ValueError(f'Invalid {name} feature tensors or labels')
            keys = [r['key'] for r in partition['records']]
            if len(set(keys)) != len(keys):
                raise ValueError('Duplicate split IDs')
            split_keys.append(set(keys))
        if split_keys[0] != reference_keys or split_keys[0] & split_keys[1]:
            raise ValueError('Invalid training reference or overlapping partitions')
        if [r['target'] for r in state['reference']] != state['train']['targets'].tolist() or [r['key'] for r in state['reference']] != [r['key'] for r in state['train']['records']]:
            raise ValueError('Reference order/labels do not match training features')
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError('Malformed preprocessed MMseqs2 dataset') from error
    return state, digest
