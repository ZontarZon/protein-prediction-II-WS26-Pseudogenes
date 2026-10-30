"""Streaming record adapters for dataset inference, independent of UI previews."""
import csv
import json
import re
from pathlib import Path

FASTA = ('.fa', '.fasta', '.faa', '.fas', '.fa.gz', '.fasta.gz', '.faa.gz', '.fas.gz')
IMAGES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
ADAPTERS = ('fasta', 'jsonl', 'csv', 'images')


def paths(root, adapter):
    suffixes = FASTA if adapter == 'fasta' else IMAGES if adapter == 'images' else ('.jsonl', '.ndjson') if adapter == 'jsonl' else ('.csv', '.tsv')
    root = Path(root).resolve()
    return sorted(p for p in root.rglob('*') if p.is_file() and p.name.lower().endswith(suffixes) and root in p.resolve().parents)


def records(root, adapter):
    if adapter not in ADAPTERS:
        raise ValueError('Select a supported record format')
    files = paths(root, adapter)
    if not files:
        raise ValueError(f'No {adapter} files found in this dataset')
    for path in files:
        source = path.relative_to(root).as_posix()
        if adapter == 'fasta':
            from ModelsAndTrainers.DataLoaders.FastaDataLoader import read_fasta
            iterator = read_fasta(path)
        elif adapter == 'jsonl':
            def lines():
                with path.open(encoding='utf-8-sig') as stream:
                    for line in stream:
                        if line.strip():
                            value = json.loads(line)
                            if not isinstance(value, dict):
                                raise ValueError('JSON Lines records must be objects')
                            yield value
            iterator = lines()
        elif adapter == 'csv':
            def rows():
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    yield from csv.DictReader(stream, delimiter='\t' if path.suffix.lower() == '.tsv' else ',')
            iterator = rows()
        else:
            iterator = iter([{'file': source, 'label': path.parent.relative_to(root).as_posix()}])
        for index, value in enumerate(iterator, 1):
            if adapter == 'fasta':
                labels = re.findall(r'(?<!\S)PE=([^\s]*)', value['header'])
                value['class'] = 'real' if len(labels) == 1 and labels[0] in {'1', '2', '3', '4'} else 'fake' if labels == ['5'] else 'unknown'
            yield {'source': source, 'source_index': index, 'data': value}


def group(record, class_field):
    value = record['data'].get(class_field) if class_field else 'all'
    if isinstance(value, (dict, list)):
        raise ValueError('Class field must contain scalar values')
    return 'unknown' if value is None else str(value)


def schema(root, adapter, class_field):
    from collections import Counter
    fields = set()
    counts = Counter()
    total = 0
    for record in records(root, adapter):
        fields.update(record['data'])
        counts[group(record, class_field)] += 1
        total += 1
        if len(counts) > 1000:
            raise ValueError('More than 1,000 classes; choose another class field or leave it empty')
    return {'fields': sorted(fields), 'total': total,
            'groups': [{'key': key, 'count': count} for key, count in sorted(counts.items())]}


def model_inputs(record, configuration, parameters, root):
    """Explicit mappings win over shared settings; never truncate sequence data."""
    import base64
    import mimetypes
    result = dict(configuration.get('inputs') or {})
    for parameter in parameters:
        key = parameter['key']
        field = configuration.get('mapping', {}).get(key)
        if not field:
            continue
        if field not in record['data']:
            raise ValueError(f'Missing mapped field {field!r} for {key}')
        value = record['data'][field]
        kind = parameter.get('type')
        if configuration['adapter'] == 'images' and field == 'file' and kind == 'file':
            path = (root / value).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise ValueError('Dataset image is unavailable')
            value = f'data:{mimetypes.guess_type(path.name)[0] or "application/octet-stream"};base64,' + base64.b64encode(path.read_bytes()).decode()
        elif kind in {'integer', 'float'} and isinstance(value, str):
            value = int(value) if kind == 'integer' else float(value)
        elif kind == 'boolean' and isinstance(value, str):
            if value.lower() not in {'true', 'false', '1', '0'}:
                raise ValueError(f'{field} must be a boolean')
            value = value.lower() in {'true', '1'}
        result[key] = value
    return result
