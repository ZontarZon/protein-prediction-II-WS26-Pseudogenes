import ast
import json
from pathlib import Path
from flask import current_app

def root():
    return current_app.config["MODEL_TRAINING_ROOT"]

def directory(key, fallback):
    return current_app.config.get(key, root() / fallback)


def dataset_records():
    from backend.dataset_registry import dataset_catalog
    roots = current_app.config.get("DATASET_ROOTS", [root() / "Datasets"])
    return dataset_catalog(roots)


def safe_child(base: Path, name: str):
    candidate = (base / name).resolve()
    return candidate if candidate == base.resolve() or base.resolve() in candidate.parents else None

def files(path: Path):
    return sorted(path.iterdir(), key=lambda p: p.name.lower()) if path.is_dir() else []

def read_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None

def model_records():
    base = directory("MODELS_DIR", "Models")
    registry = current_app.config.get("MODEL_REGISTRY")
    if registry:
        records = registry.catalog()
    else:
        records = [{"id": p.stem, "name": p.stem, "path": str(p), "type": p.suffix.lstrip("."), "interfaces": [], "inference_available": False} for p in files(base) if p.is_file() and p.suffix.lower() in {".py", ".pt", ".pth", ".onnx", ".safetensors"}]
    manager = current_app.config.get("RUN_MANAGER")
    committed = manager.committed_checkpoints() if manager else []
    for record in records:
        record["checkpoints"] = [{"id": item["id"], "name": item.get("name", item["id"]),
                                  "metadata": {key: item.get(key) for key in ("run_id", "run_name", "checkpoint", "created_at", "epoch", "step", "loss", "trainer", "datasets", "hyperparameters") if item.get(key) is not None}}
                                 for item in committed if item.get("model", "").lower() == record["id"].lower()]
    return records

def source_structure(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [{"name": node.name, "kind": "class" if isinstance(node, ast.ClassDef) else "function", "bases": [ast.unparse(base) for base in node.bases] if isinstance(node, ast.ClassDef) else [], "line": node.lineno} for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    except (OSError, SyntaxError, UnicodeError):
        return []
