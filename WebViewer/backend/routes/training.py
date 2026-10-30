from flask import Blueprint, jsonify
from .common import directory, files, read_json, safe_child

training_bp = Blueprint("training", __name__)

@training_bp.get("/runs/<path:run_id>")
def run_details(run_id):
    folder, _, checkpoint = run_id.partition("/")
    base = safe_child(directory("LEGACY_CHECKPOINTS_DIR", "ModelCechkpoints"), folder)
    if base is None or not checkpoint:
        return jsonify({"error": "Invalid run"}), 400
    selected = safe_child(base, checkpoint + ".json")
    if selected is None or not selected.is_file():
        selected = safe_child(base, checkpoint)
        selected = selected.with_suffix(".json") if selected else None
    metadata = read_json(selected) if selected and selected.is_file() else None
    if metadata is None:
        return jsonify({"error": "Run metrics not found"}), 404
    sidecars = [p for p in files(base) if p.suffix.lower() == ".json"]
    sidecars.sort(key=lambda p: ((read_json(p) or {}).get("step", (read_json(p) or {}).get("epoch", (read_json(p) or {}).get("iteration", 0))), p.name))
    series = {}
    for item in sidecars:
        row = read_json(item)
        if not isinstance(row, dict):
            continue
        step_key = next((key for key in ("step", "epoch", "iteration") if key in row), None)
        x = row.get(step_key, len(series)) if step_key else 0
        for key, value in row.items():
            if key != step_key and isinstance(value, (int, float)):
                series.setdefault(key, []).append({"x": x, "y": value})
    metrics = [{"name": key, "points": points} for key, points in series.items()]
    return jsonify({"id": run_id, "metadata": metadata, "metrics": metrics})
