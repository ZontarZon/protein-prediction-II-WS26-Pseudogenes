"""Dataset processing API; model/dataset discovery stays shared with inference."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import zipfile

from flask import Blueprint, current_app, jsonify, request, send_file

from backend.processing_data import ADAPTERS, schema
from backend.processing_manager import ACTIVE
from .common import dataset_records, safe_child

processing_bp = Blueprint('processing', __name__)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='processing-schema')
_jobs = {}
_lock = Lock()


def manager():
    return current_app.config['PROCESSING_MANAGER']


def with_urls(run_id, record_id, output):
    if output and output.get('file'):
        return {**output, 'url': f'/api/processing/runs/{run_id}/records/{record_id}/file'}
    return output


@processing_bp.get('/processing/datasets/<dataset_id>/schema')
def processing_schema(dataset_id):
    root = dataset_records().get(dataset_id)
    if root is None:
        return jsonify(error='Dataset not found'), 404
    adapter = request.args.get('adapter', 'fasta')
    class_field = request.args.get('class_field', '')
    if adapter not in ADAPTERS:
        return jsonify(error='Unsupported record format'), 400
    # Cache by file metadata; full record scans run off the HTTP thread.
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob('*')) if p.is_file() and root.resolve() in p.resolve().parents)
    key = (str(root), adapter, class_field, signature)
    with _lock:
        future = _jobs.get(key)
        if future is None:
            if sum(not f.done() for f in _jobs.values()) >= 4:
                return jsonify(status='busy')
            future = _executor.submit(schema, root, adapter, class_field)
            _jobs[key] = future
        for old in list(_jobs):
            if len(_jobs) <= 24:
                break
            if old != key and _jobs[old].done():
                del _jobs[old]
    if not future.done():
        return jsonify(status='scanning')
    try:
        return jsonify(status='ready', **future.result())
    except Exception as error:
        return jsonify(status='error', error=str(error))


@processing_bp.route('/processing/runs', methods=['GET', 'POST'])
def processing_runs():
    if request.method == 'GET':
        return jsonify(runs=manager().list())
    try:
        return jsonify(manager().start(request.get_json(silent=True) or {})), 201
    except (ValueError, TypeError, KeyError) as error:
        return jsonify(error=str(error)), 400


@processing_bp.get('/processing/runs/<run_id>')
def processing_run(run_id):
    state = manager().get(run_id)
    return jsonify(state) if state else (jsonify(error='Processing run not found'), 404)


@processing_bp.post('/processing/runs/<run_id>/stop')
def stop_processing(run_id):
    try:
        return jsonify(manager().stop(run_id))
    except ValueError as error:
        return jsonify(error=str(error)), 409


@processing_bp.get('/processing/runs/<run_id>/records')
def processing_records(run_id):
    try:
        page = max(1, int(request.args.get('page', 1)))
        size = min(100, max(1, int(request.args.get('page_size', 50))))
    except ValueError:
        return jsonify(error='Pagination values must be integers'), 400
    result = manager().page(run_id, page, size)
    if result is None:
        return jsonify(error='Processing run not found'), 404
    for item in result['items']:
        item['output'] = with_urls(run_id, item['id'], item['output'])
    return jsonify(result)


@processing_bp.get('/processing/runs/<run_id>/records/<int:record_id>')
def processing_record(run_id, record_id):
    record = manager().detail(run_id, record_id)
    if not record:
        return jsonify(error='Selected record not found'), 404
    return jsonify({**record, 'output': with_urls(run_id, record_id, record.get('output'))})


@processing_bp.get('/processing/runs/<run_id>/records/<int:record_id>/file')
def processing_record_file(run_id, record_id):
    record = manager().detail(run_id, record_id)
    file = ((record or {}).get('output') or {}).get('file')
    path = safe_child(manager().folder(run_id), file) if file else None
    return send_file(path) if path and path.is_file() else (jsonify(error='Output file not found'), 404)


@processing_bp.get('/processing/runs/<run_id>/export/<kind>')
def export_processing(run_id, kind):
    state = manager().get(run_id)
    if not state:
        return jsonify(error='Processing run not found'), 404
    if state['status'] in ACTIVE:
        return jsonify(error='Wait for processing to finish or stop it before exporting'), 409
    folder = manager().folder(run_id)
    if kind == 'jsonl':
        return send_file(folder / 'results.jsonl', as_attachment=True, download_name=f'{run_id}.jsonl', mimetype='application/x-ndjson')
    if kind == 'manifest':
        return send_file(folder / 'manifest.json', as_attachment=True, download_name=f'{run_id}-manifest.json')
    if kind == 'zip':
        # Archive only script-facing artifacts, never UI state or database internals.
        with manager().lock:
            archive = folder / 'export.zip'
            if not archive.exists():
                temporary = folder / 'export.tmp'
                with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as output:
                    for path in [folder / 'manifest.json', folder / 'results.jsonl', *sorted((folder / 'outputs').glob('*'))]:
                        if path.is_file():
                            output.write(path, path.relative_to(folder).as_posix())
                temporary.replace(archive)
        return send_file(archive, as_attachment=True, download_name=f'{run_id}.zip')
    return jsonify(error='Unknown export format'), 404
