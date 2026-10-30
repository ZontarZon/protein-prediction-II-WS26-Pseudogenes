"""Validated startup configuration; relative paths are based on the project root."""
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    'backend_host': '0.0.0.0', 'backend_port': 25565,
    'frontend_source_dir': 'TrainerView', 'frontend_static_dir': 'TrainerView/dist',
    'models_dir': 'ModelsAndTrainers/Models',
    'trainers_dir': 'ModelsAndTrainers/Trainers',
    'dataloaders_dir': 'ModelsAndTrainers/DataLoaders',
    'dataset_roots': ['ModelsAndTrainers/Datasets'],
    'dataset_viewers_dir': 'ModelsAndTrainers/Datasets/Viewers',
    'outputs_dir': 'ModelsAndTrainers/ModelOutputs',
    'legacy_checkpoints_dir': 'ModelsAndTrainers/ModelCechkpoints',
    'cors_origins': ['*'], 'max_request_size_mb': 64,
    'metric_persist_interval_seconds': 2.0, 'metric_persist_interval_steps': 50,
    'recent_metric_limit': 1000,
    'frontend_dev_host': '0.0.0.0', 'frontend_dev_port': 5173,
    'frontend_allowed_hosts': ['localhost', 'EndMySystem'],
}


def resolve_path(value, key, project_root=PROJECT_ROOT):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{key} must be a nonempty path string')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def load_settings(path=None, project_root=PROJECT_ROOT):
    path = path or PROJECT_ROOT / 'backend/settings.json'
    try:
        supplied = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(supplied, dict):
            raise ValueError('settings must be a JSON object')
        unknown = supplied.keys() - DEFAULTS.keys()
        if unknown:
            raise ValueError(f'Unknown settings: {", ".join(sorted(unknown))}')
        settings = {**DEFAULTS, **supplied}
        for key in ('backend_host', 'frontend_dev_host'):
            if not isinstance(settings[key], str) or not settings[key].strip():
                raise ValueError(f'{key} must be a nonempty string')
        for key in ('backend_port', 'frontend_dev_port', 'metric_persist_interval_steps', 'recent_metric_limit'):
            value = settings[key]
            if type(value) is not int or value < 1 or (key.endswith('port') and value > 65535):
                raise ValueError(f'{key} must be a positive integer' + (' between 1 and 65535' if key.endswith('port') else ''))
        for key in ('max_request_size_mb', 'metric_persist_interval_seconds'):
            value = settings[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be a positive finite number')
        for key in ('dataset_roots', 'cors_origins', 'frontend_allowed_hosts'):
            if not isinstance(settings[key], list) or any(not isinstance(v, str) or not v.strip() for v in settings[key]):
                raise ValueError(f'{key} must be an array of nonempty strings')
        for key in DEFAULTS:
            if key.endswith('_dir'):
                if key == 'frontend_static_dir' and settings[key] == '':
                    settings[key] = None
                else:
                    settings[key] = resolve_path(settings[key], key, project_root)
        settings['dataset_roots'] = list(dict.fromkeys(resolve_path(v, 'dataset_roots', project_root) for v in settings['dataset_roots']))
        return settings
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Could not load {path}: {error}') from error


def configure_source_packages(settings):
    """Keep bundled sources' package imports working when their folders move."""
    import sys
    import types
    for name, path in (
        ('ModelsAndTrainers', None),
        ('ModelsAndTrainers.Models', settings['models_dir']),
        ('ModelsAndTrainers.DataLoaders', settings['dataloaders_dir']),
        ('ModelsAndTrainers.Trainers', settings['trainers_dir']),
    ):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        module.__path__ = [str(path)] if path is not None else []
        module.__package__ = name
