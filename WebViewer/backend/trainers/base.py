from abc import ABC, abstractmethod
from pathlib import Path
from threading import Condition, Event
from typing import Any, Callable


class Dependency:
    def __init__(self, kind: str, label: str, description: str = "", required: bool = True):
        self.kind = kind
        self.label = label
        self.description = description
        self.required = required
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return self if instance is None else instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def schema(self):
        result = {"key": self.name, "label": self.label, "description": self.description, "required": self.required}
        if getattr(self, "choices", None) is not None:
            result["choices"] = self.choices
        if getattr(self, "interface", None) is not None:
            result["interface"] = self.interface
        return result


class Hyperparameter(Dependency):
    def __init__(self, label: str, kind: str, default: Any = None, description: str = "", required: bool = False, **constraints):
        super().__init__("hyperparameter", label, description, required)
        self.value_type = kind
        self.default = default
        self.constraints = constraints

    def schema(self):
        return {**super().schema(), "type": self.value_type, "default": self.default,
                **({"advanced": True} if self.constraints.get("advanced") else {}),
                **({"dataset_control": True} if self.constraints.get("dataset_control") else {}),
                **{key: self.constraints[key] for key in ("minimum", "maximum") if key in self.constraints}}


def model(label: str, description: str = "", required: bool = True, choices=None, interface=None):
    dependency = Dependency("model", label, description, required)
    dependency.choices = choices
    dependency.interface = interface
    return dependency


def dataset(label: str, description: str = "", required: bool = True):
    return Dependency("dataset", label, description, required)


def hyperparameter(label: str, kind: str, default: Any = None, description: str = "", required: bool = False, **constraints):
    return Hyperparameter(label, kind, default, description, required, **constraints)


class TrainerBase(ABC):
    """Cooperative lifecycle contract for long-running trainers.

    Implementations should call ``check_control`` inside their training loop and
    ``report`` after each useful metric observation.
    """

    def __init__(self, config: dict, run_dir: Path, report: Callable[[dict], None]):
        self.config = config
        self.run_dir = Path(run_dir)
        self.report_callback = report
        self._pause_condition = Condition()
        self._paused = False
        self._stop_event = Event()
        self._status = "QUEUED"
        self._metrics: dict = {}
        self.models = {}
        for name, declaration in self.__class__.configuration_schema_declarations().items():
            group = "models" if declaration.kind == "model" else "datasets" if declaration.kind == "dataset" else "hyperparameters"
            self.__dict__[name] = config.get(group, {}).get(name, declaration.default if isinstance(declaration, Hyperparameter) else None)

    def dataset_path(self, role):
        """Resolve a declared dataset role using the backend's storage configuration."""
        configured = self.config.get("dataset_paths", {}).get(role)
        if configured:
            return Path(configured)
        return Path(self.config["data_root"]) / "Datasets" / self.config["datasets"][role]

    @abstractmethod
    def train(self):
        """Run the configured training job until completion or stop request."""

    @abstractmethod
    def save_checkpoint(self, path: Path):
        """Write a resumable checkpoint to ``path``."""

    @abstractmethod
    def load_checkpoint(self, path: Path):
        """Restore trainer and model state from ``path``."""

    def pause(self):
        with self._pause_condition:
            self._paused = True
            self._status = "PAUSED"

    def resume(self):
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
            self._status = "RUNNING"

    def stop(self):
        self._stop_event.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
            self._status = "STOPPING"

    def check_control(self):
        with self._pause_condition:
            while self._paused and not self._stop_event.is_set():
                self._pause_condition.wait(timeout=0.5)
            return not self._stop_event.is_set()

    def report(self, metrics: dict):
        self._metrics.update(metrics)
        self.report_callback(dict(metrics))

    def bind_models(self, models: dict):
        """Provide resolved model objects selected for the declared model roles."""
        self.models = dict(models)

    def get_status(self):
        return self._status

    def get_metrics(self):
        return dict(self._metrics)

    # A trainer may override this hook for record-based formats (FASTA, JSON,
    # grouped objectives, etc.). File-based generation datasets can declare
    # extensions when each file corresponds to exactly one training element.
    dataset_objectives = {}

    @classmethod
    def dataset_statistics(cls, path, role):
        objective = cls.dataset_objectives.get(role)
        if not objective:
            return None
        extensions = tuple(objective.get("extensions", ()))
        if not extensions:
            return None
        total = sum(1 for item in Path(path).rglob("*") if item.is_file() and item.name.lower().endswith(extensions))
        return {"objective": objective["type"], "total": total,
                "groups": [{"key": "all", "label": "All elements", "count": total,
                            "subset_parameter": objective.get("subset_parameter")}],
                "validation_parameter": objective.get("validation_parameter")}

    def get_device_info(self):
        """Return the device actually selected by this trainer, when known."""
        return None

    def get_performance_info(self):
        """Return effective performance settings for reproducible run metadata."""
        return None

    @property
    def stop_requested(self):
        return self._stop_event.is_set()

    @property
    def paused(self):
        return self._paused

    @classmethod
    def configuration_schema_declarations(cls):
        declarations = {}
        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if isinstance(value, Dependency):
                    declarations[name] = value
        return declarations

    @classmethod
    def configuration_schema(cls):
        fields = {"models": [], "datasets": [], "hyperparameters": []}
        for name, value in cls.configuration_schema_declarations().items():
            item = value.schema()
            item["key"] = name
            fields["models" if value.kind == "model" else "datasets" if value.kind == "dataset" else "hyperparameters"].append(item)
        return fields
