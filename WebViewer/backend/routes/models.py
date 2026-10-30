from flask import Blueprint, current_app, jsonify
from .common import directory, files, safe_child, model_records, source_structure

models_bp = Blueprint("models", __name__)

@models_bp.get("/models")
def models():
    return jsonify({"models": model_records()})

@models_bp.get("/models/<model_id>/checkpoints")
def model_checkpoints(model_id):
    manager = current_app.config.get("RUN_MANAGER")
    records = manager.committed_checkpoints(model_id) if manager else []
    return jsonify({"model": model_id, "checkpoints": records})


@models_bp.get("/models/<model_id>/weights")
def model_weights(model_id):
    manager = current_app.config.get("WEIGHT_MANAGER")
    return jsonify({"model": model_id, "weights": manager.list(model_id) if manager else []})

@models_bp.get("/models/<model_id>/structure")
def model_structure(model_id):
    path = safe_child(directory("MODELS_DIR", "Models"), model_id)
    if path is None:
        return jsonify({"error": "Invalid model"}), 400
    candidates = [p for p in files(directory("MODELS_DIR", "Models")) if p.stem.lower() == model_id.lower()]
    if not candidates:
        return jsonify({"error": "Model not found"}), 404
    p = candidates[0]
    return jsonify({"model": model_id, "source": str(p), "components": source_structure(p), "available": p.suffix == ".py"})

@models_bp.get("/models/<model_id>/runs")
def model_runs(model_id):
    manager = current_app.config.get("RUN_MANAGER")
    runs = [{"id": item["id"], "name": item.get("name"), "model": item.get("model"),
             "checkpoint": item.get("checkpoint_path"), "status": "committed",
             "metadata": item, "metrics": item.get("metrics", {})}
            for item in (manager.committed_checkpoints(model_id) if manager else [])]
    return jsonify({"runs": runs})

@models_bp.get("/checkpoints")
def checkpoints():
    manager = current_app.config.get("RUN_MANAGER")
    return jsonify({"checkpoints": manager.committed_checkpoints() if manager else []})
