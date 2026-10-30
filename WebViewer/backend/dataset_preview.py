"""Bounded previews for simple files and opt-in data loader adapters."""
import ast
import base64
import csv
import importlib.util
import io
from itertools import islice
from pathlib import Path
import sys

from backend.record_search import record_page

MAX_ROWS = 20
MAX_TEXT = 4096


def loaders(folder):
    """Read preview declarations without importing unrelated training loaders."""
    result = {}
    for path in sorted(Path(folder).glob('*.py')):
        if path.name.startswith('_'):
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PREVIEW_EXTENSIONS' for t in node.targets):
                    extensions = ast.literal_eval(node.value)
                    if isinstance(extensions, (tuple, list)) and all(isinstance(s, str) and s.startswith('.') for s in extensions):
                        result[path.stem] = (path, tuple(s.lower() for s in extensions))
        except (OSError, SyntaxError, ValueError, TypeError):
            continue
    return result


def value_node(value, depth=0):
    """Convert loader output into the same display nodes used by legacy viewers."""
    if depth > 5:
        return {'type': 'text', 'value': 'Nested value omitted (preview depth limit)'}
    if value is None or isinstance(value, (bool, int, float, str)):
        import math
        if isinstance(value, float) and not math.isfinite(value):
            value = str(value)
        if isinstance(value, str) and len(value) > MAX_TEXT:
            value = value[:MAX_TEXT] + '… [truncated]'
        return {'type': 'text', 'value': value}
    from PIL import Image
    if isinstance(value, Image.Image):
        image = value.copy()
        image.thumbnail((512, 512))
        output = io.BytesIO()
        image.convert('RGB').save(output, format='PNG')
        return {'type': 'image', 'url': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode(),
                'width': value.width, 'height': value.height}
    # Tensors and arrays stay bounded before conversion to Python values.
    if type(value).__module__.split('.')[0] in {'torch', 'numpy'} and hasattr(value, 'shape'):
        shape = list(value.shape)
        flat = value.reshape(-1)[:64]
        if hasattr(flat, 'detach'):
            flat = flat.detach().cpu()
        values = flat.tolist()
        return {'type': 'object', 'label': 'Tensor / array', 'children': [
            {'type': 'text', 'label': 'Shape', 'value': str(shape)},
            {'type': 'text', 'label': 'Dtype', 'value': str(value.dtype)},
            {'type': 'json', 'label': 'First 64 values', 'value': [value_node(v, depth + 1).get('value') for v in values]},
        ]}
    if isinstance(value, dict):
        children = []
        for key, item in islice(value.items(), 32):
            children.append({**value_node(item, depth + 1), 'label': str(key)[:256]})
        if len(value) > 32:
            children.append({'type': 'text', 'value': 'Additional fields omitted'})
        return {'type': 'object', 'children': children}
    if isinstance(value, (list, tuple)):
        return {'type': 'list', 'items': [value_node(v, depth + 1) for v in value[:32]],
                'label': f'{len(value)} values (first 32 shown)' if len(value) > 32 else 'Values'}
    return {'type': 'text', 'value': str(value)[:MAX_TEXT]}


def table(records, columns=None):
    rows = [record if isinstance(record, dict) else {'value': record} for record in records]
    columns = columns if columns is not None else list(dict.fromkeys(key for row in rows for key in islice(row, 32)))[:32]
    return {'type': 'table', 'columns': [str(c)[:256] for c in columns],
            'rows': [[value_node(row.get(column)) for column in columns] for row in rows]}


def loader_preview(path, folder, loader_id=None, offset=0, search=None):
    available = loaders(folder)
    selected = loader_id or next((name for name, (_, extensions) in available.items()
                                 if path.name.lower().endswith(extensions)), None)
    if selected is None:
        return None, list(available)
    if selected not in available:
        raise ValueError('Unknown preview loader')
    source, _ = available[selected]
    name = f'ModelsAndTrainers.DataLoaders._preview_{selected}'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    preview = getattr(module, 'preview_records', None)
    if not callable(preview):
        raise ValueError(f'{selected} must define preview_records(path)')
    records, more, columns = record_page(iter(preview(path)), offset, MAX_ROWS, search or {})
    return {'view': table(records, columns), 'loader': selected, 'offset': offset,
            'searchable': True, 'has_more': more, 'next_offset': offset + MAX_ROWS if more else None,
            'notice': 'Search checks full records across this file before pagination. Displayed values may be shortened.'}, list(available)


def simple_preview(path, offset=0, search=None):
    suffix = path.suffix.lower()
    if suffix not in {'.txt', '.csv', '.tsv', '.log', '.md'}:
        return None
    if suffix in {'.csv', '.tsv'}:
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, delimiter='\t' if suffix == '.tsv' else ',')
            header = next(reader, [])
            # Unique stable field names allow unambiguous column filters.
            columns = []
            for index, name in enumerate(header):
                label = name or f'Column {index + 1}'
                candidate = label
                while candidate in columns:
                    candidate = f'{label} ({index + 1})' if candidate == label else candidate + '_'
                columns.append(candidate)
            records = (dict(zip(columns, row)) for row in reader)
            rows, more, _ = record_page(records, offset, MAX_ROWS, search or {})
        return {'view': table(rows, columns[:32]), 'searchable': True, 'offset': offset,
                'has_more': more, 'next_offset': offset + MAX_ROWS if more else None,
                'notice': 'Search checks all rows before pagination; the first row supplies field names.'}
    with path.open('r', encoding='utf-8-sig') as stream:
        text = stream.read(65537)
    truncated = len(text) > 65536
    text = text[:65536]
    return {'view': {'type': 'text', 'label': path.name, 'value': text},
            'notice': 'Text preview truncated to 64 KiB.' if truncated else None}
