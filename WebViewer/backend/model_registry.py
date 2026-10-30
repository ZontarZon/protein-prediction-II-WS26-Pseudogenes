import importlib.util
import inspect
import re
import sys
from pathlib import Path

from backend.models.base import ModelBase


class ModelRegistry:
    def __init__(self, training_root: Path, project_root: Path, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir is not None else training_root / "Models"
        self.training_root = training_root
        self.project_root = project_root

    def _classes(self):
        models_dir = self.models_dir
        for path in (self.project_root,):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        found = {}
        for path in sorted(models_dir.glob("*.py")) if models_dir.is_dir() else []:
            if path.name.startswith("_"):
                continue
            module_name = f"modeltraining_model_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception:
                continue
            for _, candidate in inspect.getmembers(module, inspect.isclass):
                if candidate is not ModelBase and issubclass(candidate, ModelBase) and candidate.__module__ == module_name:
                    metadata = candidate.metadata()
                    found[metadata["id"]] = (candidate, path)
        return found

    def catalog(self):
        models_dir = self.models_dir
        catalog = {}
        for path in sorted(models_dir.glob("*.py")) if models_dir.is_dir() else []:
            if path.name.startswith("_"):
                continue
            catalog[path.stem] = {"id": path.stem, "name": path.stem, "description": "Discovered model source. Add ModelBase metadata for typed training and inference support.",
                                  "category": "source", "path": str(path), "type": "py",
                                  "inputs": [], "output": {"type": "numerical", "label": "Undeclared output"},
                                  "interfaces": [], "inference_parameters": [], "checkpoint_required": True,
                                  "inference_available": False, "checkpoints": []}
        for model_id, (model_type, path) in self._classes().items():
            catalog[model_id] = {**catalog.get(model_id, {}), **model_type.metadata(),
                                 "path": str(path), "type": "py", "checkpoints": []}
        return list(catalog.values())

    def get(self, model_id):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", model_id or ""):
            return None
        return self._classes().get(model_id)

    def create(self, model_id, parameters=None):
        result = self.get(model_id)
        if result is None:
            raise ValueError(f"Model {model_id} is not a registered ModelBase implementation")
        model_type, _ = result
        return model_type(**(parameters or {}))

    def inference_models(self):
        return [item for item in self.catalog() if item.get("inference_available")]
