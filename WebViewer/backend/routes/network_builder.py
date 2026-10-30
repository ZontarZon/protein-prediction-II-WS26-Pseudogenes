import json
import re
from flask import Blueprint, current_app, request, jsonify
from .common import directory
from backend.network_builder import catalog, analyze, source, import_sequential

network_builder_bp = Blueprint('network_builder', __name__)

@network_builder_bp.get('/network-builder/catalog')
def operations():
    import torch
    return jsonify({'version': torch.__version__, 'items': catalog()})

@network_builder_bp.post('/network-builder/analyze')
def validate():
    try:
        return jsonify(analyze(request.get_json()))
    except Exception as exc:
        return jsonify(error=str(exc)), 400

@network_builder_bp.post('/network-builder/export')
def export():
    data = request.get_json() or {}
    graph, name = data.get('graph', {}), data.get('name', '')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}', name):
        return jsonify(error='Use a model name starting with a letter, containing only letters, numbers and underscores (max 80 characters)'), 400
    folder = directory("MODELS_DIR", "Models")
    py, document = folder / f'{name}.py', folder / f'{name}.graph.json'
    if py.exists() or document.exists():
        return jsonify(error='A model with this name already exists. Choose a new name.'), 409
    try:
        metrics = analyze(graph)
        code = source(graph, name)
        compile(code, str(py), 'exec')
        folder.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement, including concurrent exports.
        with py.open('x', encoding='utf-8') as handle:
            handle.write(code)
        try:
            with document.open('x', encoding='utf-8') as handle:
                json.dump(graph, handle, indent=2, allow_nan=False)
        except Exception:
            py.unlink()
            raise
        return jsonify(path=str(py), model=name, metrics=metrics), 201
    except Exception as exc:
        return jsonify(error=str(exc)), 400

@network_builder_bp.get('/network-builder/models')
def models():
    return jsonify(models=[p.stem for p in sorted((directory("MODELS_DIR", "Models")).glob('*.py')) if not p.name.startswith('_')])

@network_builder_bp.post('/network-builder/import')
def load():
    data = request.get_json() or {}
    name = data.get('model', '')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        return jsonify(error='Invalid model name'), 400
    path = directory("MODELS_DIR", "Models") / f'{name}.py'
    if not path.is_file(): return jsonify(error='Model not found'), 404
    document = path.with_suffix('.graph.json')
    try:
        if document.is_file(): return jsonify(graph=json.loads(document.read_text()))
        return jsonify(import_sequential(path, data.get('class_name')))
    except Exception as exc:
        return jsonify(error=str(exc)), 400
