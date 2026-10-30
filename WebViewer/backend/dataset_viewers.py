"""Discovery and generic rendering for dataset inspection plugins."""
import importlib.util
import inspect
import base64
import io
import sys
from pathlib import Path
from PIL import Image


class DatasetViewer:
    dataset_id = None

    def get_data_layout(self, sample=None):
        raise NotImplementedError

    def get_data_view(self, data, dataset_id=None, entry_id=None):
        raise NotImplementedError


class FallbackDatasetViewer(DatasetViewer):
    def get_data_layout(self, sample=None):
        return infer_layout(sample) if sample is not None else {"type": "object", "label": "Entry", "children": []}

    def get_data_view(self, data, dataset_id=None, entry_id=None):
        return generic_view(data, dataset_id, entry_id)


def infer_layout(value, label="Entry"):
    if isinstance(value, dict):
        return {"type": "object", "label": label, "children": [infer_layout(item, str(key)) for key, item in value.items()]}
    if isinstance(value, list):
        sample = value[0] if value else None
        return {"type": "list", "label": label, "item": infer_layout(sample, "Item") if sample is not None else {"type": "unknown", "label": "Item"}}
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, (int, float)):
        kind = "number"
    elif isinstance(value, str):
        kind = "string"
    elif value is None:
        kind = "null"
    else:
        kind = "unknown"
    return {"type": kind, "label": label}


def generic_view(value, dataset_id=None, entry_id=None, key=""):
    if isinstance(value, dict):
        return {"type": "object", "label": key or "Entry", "children": [generic_view(item, dataset_id, entry_id, str(name)) for name, item in value.items()]}
    if isinstance(value, list):
        return {"type": "list", "label": key, "items": [generic_view(item, dataset_id, entry_id, "Item") for item in value]}
    if isinstance(value, bool):
        return {"type": "boolean", "label": key, "value": value}
    if isinstance(value, (int, float)) or value is None:
        return {"type": "number" if value is not None else "null", "label": key, "value": value}
    if isinstance(value, str):
        if "blob" in key.lower() or "image" in key.lower():
            try:
                if value.startswith("data:"):
                    value = value.split(",", 1)[1]
                Image.open(io.BytesIO(base64.b64decode(value, validate=True))).verify()
                return {"type": "image", "label": key, "url": f"/api/datasets/{dataset_id}/entries/{entry_id}/blobs/{key}"}
            except Exception:
                return {"type": "file", "label": key, "value": "Image blob is missing or invalid."}
        return {"type": "text", "label": key, "value": value}
    return {"type": "json", "label": key, "value": str(value)}


def discover_viewer(dataset_id, viewers_root: Path):
    for path in sorted(viewers_root.glob("*.py")) if viewers_root.is_dir() else []:
        if path.name.startswith("_"):
            continue
        module_name = f"dataset_viewer_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if not spec or not spec.loader:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            for _, candidate in inspect.getmembers(module, inspect.isclass):
                if candidate is not DatasetViewer and issubclass(candidate, DatasetViewer) and candidate.__module__ == module_name and candidate.dataset_id == dataset_id:
                    return candidate()
        except Exception:
            continue
    return FallbackDatasetViewer()
