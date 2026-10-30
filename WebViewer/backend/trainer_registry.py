import importlib.util
import inspect
import re
import sys
from pathlib import Path

from backend.trainers.base import TrainerBase


class TrainerRegistry:
    def __init__(self, training_root: Path, project_root: Path, trainers_dir=None):
        self.trainers_dir = Path(trainers_dir) if trainers_dir is not None else training_root / "Trainers"
        self.training_root = training_root
        self.project_root = project_root

    def _classes(self):
        trainers_dir = self.trainers_dir
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        found = {}
        for path in sorted(trainers_dir.glob("*.py")) if trainers_dir.is_dir() else []:
            if path.name.startswith("_"):
                continue
            module_name = f"modeltraining_trainer_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception:
                continue
            for name, candidate in inspect.getmembers(module, inspect.isclass):
                if candidate is not TrainerBase and issubclass(candidate, TrainerBase) and candidate.__module__ == module_name and not inspect.isabstract(candidate):
                    found[name] = candidate
        return found

    def list(self):
        return [{"id": name, "name": name, "schema": trainer.configuration_schema()} for name, trainer in self._classes().items()]

    def get(self, trainer_id: str):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", trainer_id):
            return None
        return self._classes().get(trainer_id)
