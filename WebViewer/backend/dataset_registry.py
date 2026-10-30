"""Discover dataset directories across configured storage roots without reading files."""
from pathlib import Path


def dataset_catalog(roots):
    datasets = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Dataset root does not exist or is not a directory: {root}')
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name == 'Viewers':
                continue
            resolved = path.resolve()
            # Do not expose symlinks escaping a configured root.
            if root not in resolved.parents:
                continue
            if path.name in datasets and datasets[path.name] != resolved:
                raise ValueError(f'Duplicate dataset name {path.name!r} in configured dataset roots; rename one of the folders')
            datasets[path.name] = resolved
    return datasets
