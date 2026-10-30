"""Persistent, paged dataset inference with clean per-record JSONL exports."""
import json
import random
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from backend.dataset_registry import dataset_catalog
from backend.processing_data import ADAPTERS, records, group, model_inputs

ACTIVE = {'QUEUED', 'INDEXING', 'RUNNING', 'STOPPING'}


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


class ProcessingManager:
    def __init__(self, inference, dataset_roots, outputs_dir):
        self.inference = inference
        self.dataset_roots = dataset_roots
        self.root = Path(outputs_dir) / 'processing_runs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stops = {}
        for folder in self.root.iterdir():
            state = self.get(folder.name)
            if state and state['status'] in ACTIVE:
                self.update(folder.name, status='FAILED', finished_at=time.time(), error='Backend restarted; completed records remain available.')

    def folder(self, run_id):
        if not isinstance(run_id, str) or not run_id or any(c not in '0123456789abcdef' for c in run_id):
            return None
        path = self.root / run_id
        return path if path.parent.resolve() == self.root.resolve() else None

    def get(self, run_id):
        folder = self.folder(run_id)
        try:
            with self.lock:
                return json.loads((folder / 'state.json').read_text()) if folder else None
        except (OSError, ValueError):
            return None

    def update(self, run_id, **changes):
        with self.lock:
            state = self.get(run_id) or {}
            state.update(changes)
            path = self.folder(run_id) / 'state.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, allow_nan=False))
            temporary.replace(path)
            return state

    def list(self):
        return sorted(filter(None, (self.get(p.name) for p in self.root.iterdir() if p.is_dir())),
                      key=lambda s: s.get('created_at', 0), reverse=True)

    def validate(self, config):
        if not isinstance(config, dict):
            raise ValueError('Configuration must be an object')
        root = dataset_catalog(self.dataset_roots).get(config.get('dataset', ''))
        if root is None:
            raise ValueError('Select an available dataset')
        if config.get('adapter') not in ADAPTERS:
            raise ValueError('Select a supported record format')
        registered = self.inference.model_registry.get(config.get('model', ''))
        if not registered or not registered[0].has_inference():
            raise ValueError('Select an inference model')
        parameters = registered[0].metadata()['inference_parameters']
        allowed = {p['key'] for p in parameters}
        for key in ('inputs', 'mapping', 'subsets'):
            if not isinstance(config.get(key, {}), dict):
                raise ValueError(f'{key} must be an object')
        if set(config.get('mapping', {})) - allowed or not all(isinstance(v, str) for v in config.get('mapping', {}).values()):
            raise ValueError('Invalid input field mapping')
        if callable(getattr(registered[0], 'prepare_inference', None)) and config.get('mapping', {}).get('device'):
            raise ValueError('Device must be a shared setting for prepared inference')
        for count in config.get('subsets', {}).values():
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise ValueError('Subset counts must be nonnegative integers (0 = all), or null to exclude a class')
        if not isinstance(config.get('class_field', ''), str):
            raise ValueError('Class field must be a string')
        seed = config.get('seed', 7)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('Selection seed must be an integer')
        sample = next(records(root, config['adapter']), None)
        if not sample:
            raise ValueError('Dataset contains no records')
        supplied = model_inputs(sample, config, parameters, root)
        errors = self.inference.validate({'model': config['model'], 'inputs': supplied, 'checkpoint': config.get('checkpoint')})
        if errors:
            raise ValueError('; '.join(errors))
        return root, parameters

    def start(self, config):
        self.validate(config)
        with self.lock:
            if any(s['status'] in ACTIVE for s in self.list()):
                raise ValueError('Wait for the current data processing job to finish, or stop it first')
            run_id = uuid.uuid4().hex
            folder = self.folder(run_id)
            folder.mkdir()
            (folder / 'records').mkdir()
            # Configuration is stored once, outside the record export.
            (folder / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'configuration': config}, indent=2))
            with sqlite3.connect(folder / 'records.sqlite') as db:
                db.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, class TEXT, priority REAL, selected INTEGER DEFAULT 0, record TEXT, status TEXT DEFAULT "PENDING", summary TEXT, error TEXT)')
                db.execute('CREATE INDEX selected_records ON items(selected,id)')
                db.execute('CREATE INDEX class_selection ON items(class,priority)')
            (folder / 'results.jsonl').touch()
            state = self.update(run_id, id=run_id, configuration=config, status='QUEUED', created_at=time.time(),
                                indexed=0, total=0, completed=0, failed=0, groups={})
            event = threading.Event()
            self.stops[run_id] = event
            threading.Thread(target=self.execute, args=(run_id, config, event), daemon=True, name=f'processing-{run_id[:8]}').start()
            return state

    def _prepare_data(self, destination, root, config, event, run_id):
        """Stream into local scratch SQLite, then publish only selected records.

        Like ProteinPETrainer, build lookup indexes after the scan and commit
        the selected snapshot once. Avoid repeatedly journaling a growing index
        on the output mount. Scratch storage is closed and removed on all exits.
        """
        rng = random.Random(config.get('seed', 7))
        counts = {}
        try:
            with TemporaryDirectory(prefix='dnatrainer-processing-') as temporary:
                with closing(sqlite3.connect(Path(temporary) / 'index.sqlite')) as scratch:
                    scratch.execute('PRAGMA cache_size=-8192')
                    scratch.execute('CREATE TABLE candidates (id INTEGER PRIMARY KEY, class TEXT, priority REAL, record TEXT)')
                    scratch.set_progress_handler(lambda: int(event.is_set()), 1000)
                    for index, record in enumerate(records(root, config['adapter']), 1):
                        if event.is_set():
                            raise InterruptedError()
                        label = group(record, config.get('class_field', ''))
                        counts[label] = counts.get(label, 0) + 1
                        scratch.execute('INSERT INTO candidates VALUES (?,?,?,?)',
                                        (index, label, rng.random(), json.dumps(record, allow_nan=False)))
                        if index % 1000 == 0:
                            self.update(run_id, indexed=index)
                    self.update(run_id, indexed=sum(counts.values()))
                    scratch.execute('CREATE INDEX candidate_classes ON candidates(class,priority)')
                    scratch.commit()
                    if event.is_set():
                        raise InterruptedError()
                    with destination:
                        destination.set_progress_handler(lambda: int(event.is_set()), 1000)
                        for label in counts:
                            limit = config.get('subsets', {}).get(label, 0)
                            if limit is None:
                                continue
                            selected = scratch.execute('SELECT id,class,priority,record FROM candidates WHERE class=? ORDER BY priority LIMIT ?',
                                                       (label, limit or -1))
                            destination.executemany('INSERT INTO items(id,class,priority,record,selected) VALUES (?,?,?,?,1)', selected)
                        if event.is_set():
                            raise InterruptedError()
                    destination.set_progress_handler(None, 0)
                    total = destination.execute('SELECT COUNT(*) FROM items WHERE selected=1').fetchone()[0]
                    if total == 0:
                        raise ValueError('No records selected; include at least one class')
            return counts, total
        except sqlite3.OperationalError:
            if event.is_set():
                raise InterruptedError() from None
            raise
        finally:
            destination.set_progress_handler(None, 0)

    def execute(self, run_id, config, event):
        folder = self.folder(run_id)
        completed = failed = 0
        try:
            root, parameters = self.validate(config)
            self.update(run_id, status='INDEXING')
            with closing(sqlite3.connect(folder / 'records.sqlite', timeout=30)) as db:
                counts, total = self._prepare_data(db, root, config, event, run_id)
                self.update(run_id, status='RUNNING', indexed=sum(counts.values()), total=total, groups=counts)
                if event.is_set():
                    raise InterruptedError()
                model, checkpoint = self.inference.create_model(config)
                prepared = callable(getattr(model, 'prepare_inference', None)) and callable(getattr(model, 'infer_prepared', None))
                if prepared:
                    model.prepare_inference(config.get('inputs', {}), str(checkpoint))
                output_type = model.metadata()['output']['type']
                with (folder / 'results.jsonl').open('a', encoding='utf-8') as export:
                    cursor = db.execute('SELECT id,record FROM items WHERE selected=1 ORDER BY id')
                    for record_id, raw in cursor:
                        if event.is_set():
                            raise InterruptedError()
                        record = json.loads(raw)
                        inputs = None
                        output = None
                        error = None
                        def progress(step, total_steps, output=None, **kwargs):
                            if event.is_set():
                                raise InterruptedError()
                            # Avoid writing large metadata on every integration step.
                        try:
                            inputs = model_inputs(record, config, parameters, root)
                            value = model.infer_prepared(inputs, progress_callback=progress) if prepared else model.infer(inputs, str(checkpoint), progress_callback=progress)
                            if event.is_set():
                                raise InterruptedError()
                            output = self.save_output(folder, record_id, output_type, value)
                        except InterruptedError:
                            raise
                        except Exception as exception:
                            error = str(exception)
                        result = {'record_id': record_id, **record, 'inputs': inputs, 'output': output, 'error': error}
                        encoded = json.dumps(result, allow_nan=False, ensure_ascii=False)
                        path = folder / 'records' / f'{record_id}.json'
                        path.write_text(encoded, encoding='utf-8')
                        export.write(encoded + '\n')
                        export.flush()
                        summary = self.summary(output)
                        db.execute('UPDATE items SET status=?,summary=?,error=? WHERE id=?',
                                   ('FAILED' if error else 'COMPLETED', json.dumps(summary), error, record_id))
                        db.commit()
                        completed += 1
                        failed += int(error is not None)
                        self.update(run_id, completed=completed, failed=failed)
            self.update(run_id, status='COMPLETED_WITH_ERRORS' if failed else 'COMPLETED', finished_at=time.time())
        except InterruptedError:
            self.update(run_id, status='STOPPED', finished_at=time.time())
        except Exception as exception:
            self.update(run_id, status='FAILED', error=str(exception), finished_at=time.time())
        finally:
            with self.lock:
                self.stops.pop(run_id, None)

    def save_output(self, folder, record_id, kind, value):
        if kind == 'image':
            # Use the same tensor/PIL conversion as single-record inference.
            class Storage:
                _folder = lambda self, _: folder
            from backend.inference_manager import InferenceRunManager
            result = InferenceRunManager._save_intermediate(Storage(), '', value, str(record_id))
            if result['type'] != 'image':
                raise ValueError('Image model did not return an image')
            return result
        value = json_value(value)
        json.dumps(value, allow_nan=False)
        return {'type': kind, 'value': value}

    @staticmethod
    def summary(output):
        if not output:
            return None
        if output['type'] == 'classification':
            return {'type': 'classification', 'value': {'classes': output.get('value', {}).get('classes', [])}}
        if output['type'] == 'image':
            return output
        value = output.get('value')
        if isinstance(value, (str, dict, list)):
            # The table stays compact; complete values are always in the detail/export.
            value = json.dumps(value, ensure_ascii=False)[:160] if not isinstance(value, str) else value[:160]
        return {'type': output['type'], 'value': value}

    def page(self, run_id, page=1, size=50):
        state = self.get(run_id)
        if state is None:
            return None
        folder = self.folder(run_id)
        with sqlite3.connect(folder / 'records.sqlite', timeout=30) as db:
            total = db.execute('SELECT COUNT(*) FROM items WHERE selected=1').fetchone()[0]
            rows = []
            for key, label, raw, status, summary, error in db.execute('SELECT id,class,record,status,summary,error FROM items WHERE selected=1 ORDER BY id LIMIT ? OFFSET ?', (size, (page - 1) * size)):
                record = json.loads(raw)
                data = record['data']
                title = str(data.get('id', data.get('file', f"{record['source']}:{record['source_index']}")))
                preview = str(data.get('sequence', data.get('header', data)))[:150]
                rows.append({'id': key, 'label': title, 'class': label, 'preview': preview, 'status': status,
                             'output': json.loads(summary) if summary else None, 'error': error})
        return {'items': rows, 'total': total, 'page': page, 'page_size': size}

    def detail(self, run_id, record_id):
        if not self.get(run_id):
            return None
        folder = self.folder(run_id)
        path = folder / 'records' / f'{record_id}.json'
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
        with sqlite3.connect(folder / 'records.sqlite') as db:
            row = db.execute('SELECT record,error FROM items WHERE id=? AND selected=1', (record_id,)).fetchone()
            return {'record_id': record_id, **json.loads(row[0]), 'output': None, 'error': row[1]} if row else None

    def stop(self, run_id):
        with self.lock:
            event = self.stops.get(run_id)
            if not event or (self.get(run_id) or {}).get('status') not in ACTIVE:
                raise ValueError('This processing job is not active')
            self.update(run_id, status='STOPPING')
            event.set()
            return self.get(run_id)
