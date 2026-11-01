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
