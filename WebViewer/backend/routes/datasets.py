import base64
import io
import json
import math
from flask import Blueprint, jsonify, request, send_file
from PIL import Image, UnidentifiedImageError

from backend.dataset_viewers import discover_viewer
from backend.record_search import parse_search
from backend.dataset_preview import loader_preview, simple_preview, loaders
from .common import directory, dataset_records, safe_child

datasets_bp = Blueprint("datasets", __name__)


def _dataset(dataset_id):
    return dataset_records().get(dataset_id)


def _viewer(dataset_id):
    return discover_viewer(dataset_id, directory("DATASET_VIEWERS_DIR", "Datasets/Viewers"))


def _entries(path):
    return sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix().lower())


def _entry_path(dataset_path, entry_id):
    path = safe_child(dataset_path, entry_id)
    return path if path and path.is_file() else None


def _raw_entry(path):
    if path.suffix.lower() != ".json":
        return {"_file": path.name}
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("JSON exceeds the 1 MiB preview limit; select a streaming preview loader")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, (dict, list, str, int, float, bool)) and value is not None:
            raise ValueError("Unsupported JSON value")
        return value
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Could not read JSON entry {path.name}: {error}") from error


@datasets_bp.get("/datasets")
def datasets():
    return jsonify({"datasets": [{"id": name, "name": name,
                                  "itemCount": len(_entries(path))}
                                 for name, path in dataset_records().items()]})


@datasets_bp.get("/datasets/<dataset_id>/statistics")
def dataset_statistics_view(dataset_id):
    path = _dataset(dataset_id)
    if path is None:
        return jsonify({"error": "Dataset not found or invalid"}), 404
    from backend.dataset_statistics import DatasetInventory, dataset_statistics
    return jsonify(dataset_statistics(DatasetInventory, path, 'inventory'))


@datasets_bp.get("/datasets/<dataset_id>")
@datasets_bp.get("/datasets/<dataset_id>/entries")
def dataset_items(dataset_id):
    path = _dataset(dataset_id)
    if path is None:
        return jsonify({"error": "Dataset not found or invalid"}), 404
    entries = _entries(path)
    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(500, max(1, int(request.args.get("page_size", 100))))
    except ValueError:
        return jsonify({"error": "page and page_size must be integers"}), 400
    unfiltered_total = len(entries)
    query = request.args.get("q", "").strip().casefold()
    extension = request.args.get("extension", "").strip().lower().lstrip(".")
    sort = request.args.get("sort", "name")
    descending = request.args.get("order", "asc") == "desc"
    if sort not in {"name", "size"} or request.args.get("order", "asc") not in {"asc", "desc"}:
        return jsonify({"error": "Invalid sort or order"}), 400
    try:
        minimum = float(request.args.get("min_bytes", 0))
        maximum = float(request.args["max_bytes"]) if request.args.get("max_bytes") else None
        if not math.isfinite(minimum) or minimum < 0 or (maximum is not None and (not math.isfinite(maximum) or maximum < minimum)):
            raise ValueError()
    except ValueError:
        return jsonify({"error": "File size bounds must be finite nonnegative numbers; maximum must be at least minimum"}), 400
    entries = [item for item in entries
               if query in item.relative_to(path).as_posix().casefold()
               and (not extension or item.name.lower().endswith("." + extension))
               and item.stat().st_size >= minimum
               and (maximum is None or item.stat().st_size <= maximum)]
    entries.sort(key=lambda item: (item.stat().st_size, item.relative_to(path).as_posix().casefold()) if sort == "size" else item.relative_to(path).as_posix().casefold(), reverse=descending)
    total = len(entries)
    selected = entries[(page - 1) * page_size:page * page_size]
    values = [{"name": item.relative_to(path).as_posix(), "path": item.relative_to(path).as_posix(), "size": item.stat().st_size}
              for item in selected]
    return jsonify({"dataset": dataset_id, "entries": values, "items": values, "itemCount": total,
                    "page": page, "page_size": page_size, "total": total,
                    "total_pages": (total + page_size - 1) // page_size, "unfiltered_total": unfiltered_total})


@datasets_bp.get("/datasets/<dataset_id>/layout")
def dataset_layout(dataset_id):
    path = _dataset(dataset_id)
    if path is None:
        return jsonify({"error": "Dataset not found or invalid"}), 404
    viewer = _viewer(dataset_id)
    sample = None
    for entry in _entries(path):
        if entry.suffix.lower() == ".json":
            try:
                sample = _raw_entry(entry)
                break
            except ValueError:
                continue
    try:
        return jsonify({"dataset": dataset_id, "layout": viewer.get_data_layout(sample)})
    except Exception as error:
        return jsonify({"error": f"Dataset layout could not be generated: {error}"}), 500


@datasets_bp.get("/datasets/preview-loaders")
def preview_loaders():
    return jsonify({"loaders": list(loaders(directory("DATALOADERS_DIR", "DataLoaders")))})


@datasets_bp.get("/datasets/<dataset_id>/entries/<path:entry_id>/view")
def dataset_entry_view(dataset_id, entry_id):
    dataset_path = _dataset(dataset_id)
    if dataset_path is None:
        return jsonify({"error": "Dataset not found or invalid"}), 404
    path = _entry_path(dataset_path, entry_id)
    if path is None:
        return jsonify({"error": "Dataset entry not found"}), 404
    folder = directory("DATALOADERS_DIR", "DataLoaders")
    loader_id = request.args.get("loader") or None
    try:
        offset = int(request.args.get("offset", 0))
        if offset < 0:
            raise ValueError("Preview offset must be nonnegative")
        search = parse_search(request.args.get("search"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    available = list(loaders(folder))
    result = None
    try:
        if loader_id:
            result, available = loader_preview(path, folder, loader_id, offset, search)
        elif path.suffix.lower() == ".json":
            result = {"view": _viewer(dataset_id).get_data_view(_raw_entry(path), dataset_id, entry_id)}
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
                result = {"view": {"type": "image", "label": path.name,
                          "url": f"/api/datasets/{dataset_id}/entries/{entry_id}/file"}}
            except (UnidentifiedImageError, OSError):
                result = simple_preview(path, offset, search)
                if result is None:
                    result, available = loader_preview(path, folder, offset=offset, search=search)
        if result is None:
            result = {"view": {"type": "file", "label": path.name,
                      "value": f"File · {path.stat().st_size} bytes. Select a compatible data loader to preview its records."}}
        return jsonify({"dataset": dataset_id, "entry": entry_id, "loaders": available, **result})
    except Exception as error:
        return jsonify({"error": f"Preview failed: {error}", "loaders": available}), 422



@datasets_bp.get("/datasets/<dataset_id>/entries/<path:entry_id>/blobs/<field>")
def dataset_blob(dataset_id, entry_id, field):
    dataset_path = _dataset(dataset_id)
    path = _entry_path(dataset_path, entry_id) if dataset_path else None
    if path is None or path.suffix.lower() != ".json":
        return jsonify({"error": "Dataset image entry not found"}), 404
    try:
        data = _raw_entry(path)
        encoded = data.get(field) if isinstance(data, dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("Image data is missing")
        payload = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(payload))
        image.verify()
        image = Image.open(io.BytesIO(payload))
        output = io.BytesIO()
        image.save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png", max_age=0)
    except (ValueError, OSError, UnidentifiedImageError) as error:
        return jsonify({"error": f"Image blob is unavailable or invalid: {error}"}), 422


@datasets_bp.get("/datasets/<dataset_id>/entries/<path:entry_id>/file")
def dataset_entry_file(dataset_id, entry_id):
    dataset_path = _dataset(dataset_id)
    path = _entry_path(dataset_path, entry_id) if dataset_path else None
    if path is None:
        return jsonify({"error": "Dataset file not found"}), 404
    try:
        with Image.open(path) as image:
            mime = Image.MIME.get(image.format, "application/octet-stream")
            image.verify()
        return send_file(path, mimetype=mime, max_age=0)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Dataset file is not a supported image"}), 422
