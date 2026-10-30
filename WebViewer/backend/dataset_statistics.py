"""Background, cached objective-aware dataset statistics (no model construction)."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from threading import Lock

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='dataset-statistics')
_jobs = OrderedDict()
_lock = Lock()


def dataset_statistics(trainer, path, role):
    # File metadata invalidates completed counts when data changes. Contents are
    # read only by the background trainer hook, never by the HTTP request thread.
    paths = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    key = (trainer.__module__, trainer.__name__, str(path), role, signature)
    with _lock:
        future = _jobs.get(key)
        if future is None:
            if sum(not job.done() for job in _jobs.values()) >= 8:
                return {'status': 'busy'}
            future = _executor.submit(trainer.dataset_statistics, path, role)
            _jobs[key] = future
        _jobs.move_to_end(key)
        for old_key in list(_jobs):
            if len(_jobs) <= 32:
                break
            if _jobs[old_key].done():
                del _jobs[old_key]
    if not future.done():
        return {'status': 'scanning'}
    try:
        result = future.result()
        return {'status': 'ready', **result} if result is not None else {'status': 'unsupported'}
    except Exception as error:
        return {'status': 'error', 'error': str(error)}


class DatasetInventory:
    """Dataset-browser counts independent of the selected training objective."""
    @classmethod
    def dataset_statistics(cls, path, role):
        import re
        from collections import Counter
        from ModelsAndTrainers.DataLoaders.FastaDataLoader import FastaDataset
        paths = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
        suffixes = ('.fa', '.fasta', '.faa', '.fas', '.fa.gz', '.fasta.gz', '.faa.gz', '.fas.gz')
        fasta_paths = [p for p in paths if p.name.lower().endswith(suffixes)]
        if not fasta_paths:
            return {'total': len(paths), 'unit': 'files', 'file_count': len(paths),
                    'groups': [], 'labels': [], 'note': 'File count only; record-level counts are not available for this format.'}
        counts = Counter()
        for fasta in fasta_paths:
            for record in FastaDataset(fasta):
                labels = re.findall(r'(?<!\S)PE=([^\s]*)', record['header'])
                label = labels[0] if len(labels) == 1 and labels[0] in {'1', '2', '3', '4', '5'} else 'unknown'
                counts[label] += 1
        groups = []
        if any(counts[str(level)] for level in range(1, 6)):
            groups = [{'key': 'real', 'label': 'Class 1 · Real (PE 1–4)', 'count': sum(counts[str(n)] for n in range(1, 5))},
                      {'key': 'fake', 'label': 'Class 2 · Fake (PE 5)', 'count': counts['5']}]
            if counts['unknown']:
                groups.append({'key': 'unknown', 'label': 'Missing or invalid PE label', 'count': counts['unknown']})
        names = ['Evidence at protein level', 'Evidence at transcript level', 'Inferred from homology', 'Predicted', 'Uncertain']
        return {'total': sum(counts.values()), 'unit': 'records', 'file_count': len(paths),
                'fasta_file_count': len(fasta_paths), 'groups': groups,
                'labels': [{'key': str(n), 'label': f'PE={n} · {name}', 'count': counts[str(n)]} for n, name in enumerate(names, 1)] if groups else [],
                'note': 'Counts cover every FASTA record. Real/Fake follows the PE classification objective; PE=5 means uncertain existence.' if groups else 'No PE classes were found in the FASTA headers.'}
