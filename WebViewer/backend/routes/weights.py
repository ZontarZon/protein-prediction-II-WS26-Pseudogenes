import json

from flask import Blueprint, current_app, jsonify, request

weights_bp = Blueprint("weights", __name__)


@weights_bp.get("/weights")
def list_weights():
    manager = current_app.config["WEIGHT_MANAGER"]
    model_id = request.args.get("model")
    return jsonify({"weights": manager.list(model_id)})


@weights_bp.get("/weights/<weight_id>")
def weight_detail(weight_id):
    value = current_app.config["WEIGHT_MANAGER"].get(weight_id)
    return (jsonify(value), 200) if value else (jsonify({"error": "Model weight artifact not found"}), 404)


@weights_bp.get("/checkpoints/<committed_id>")
def committed_checkpoint_detail(committed_id):
    manager = current_app.config["RUN_MANAGER"]
    record = next((item for item in manager.committed_checkpoints() if item.get("id") == committed_id), None)
    if not record:
        return jsonify({"error": "Committed checkpoint not found"}), 404
    return jsonify({**record, "models": {"model": record.get("model")}, "history": record.get("history", []),
                    "metrics": record.get("metrics", {})})


@weights_bp.delete("/checkpoints/<committed_id>")
def delete_committed_checkpoint(committed_id):
    manager = current_app.config["RUN_MANAGER"]
    try:
        result = manager.delete_committed_checkpoint(committed_id)
        return (jsonify(result), 200) if result else (jsonify({"error": "Committed checkpoint not found"}), 404)
    except ValueError as error:
        return jsonify({"error": str(error)}), 409
    except OSError as error:
        return jsonify({"error": f"Could not delete committed checkpoint: {error}"}), 500


@weights_bp.post("/weights/import")
def import_weights():
    manager = current_app.config["WEIGHT_MANAGER"]
    try:
        config = json.loads(request.form.get("model_config", "{}"))
        if not isinstance(config, dict):
            raise ValueError("Model configuration must be a JSON object")
        result = manager.import_file(request.files.get("file"), request.form.get("model", ""), config)
        return jsonify(result), 201
    except (ValueError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400


def _extract(model_id, path, source):
    manager = current_app.config["WEIGHT_MANAGER"]
    try:
        return jsonify(manager.extract_from_checkpoint(model_id, path, source)), 201
    except (ValueError, OSError) as error:
        return jsonify({"error": str(error)}), 409


@weights_bp.post("/checkpoints/<committed_id>/extract-weights")
def extract_committed_weights(committed_id):
    run_manager = current_app.config["RUN_MANAGER"]
    record = next((item for item in run_manager.committed_checkpoints() if item.get("id") == committed_id), None)
    if not record:
        return jsonify({"error": "Committed checkpoint not found"}), 404
    path = (run_manager.outputs_root / record.get("checkpoint_path", "")).resolve()
    if run_manager.outputs_root.resolve() not in path.parents or not path.is_file():
        return jsonify({"error": "Committed checkpoint data is unavailable"}), 404
    return _extract(record["model"], path, record)


@weights_bp.post("/runs/<run_id>/checkpoints/<checkpoint_id>/extract-weights")
def extract_run_weights(run_id, checkpoint_id):
    run_manager = current_app.config["RUN_MANAGER"]
    checkpoint = run_manager.checkpoint(run_id, checkpoint_id)
    run = run_manager.get(run_id) if checkpoint else None
    if not checkpoint or not run:
        return jsonify({"error": "Training checkpoint not found"}), 404
    model_id = next(iter(checkpoint.get("models") or run.get("configuration", {}).get("models", {}).values()), "")
    try:
        record, path = current_app.config["WEIGHT_MANAGER"].resolve_checkpoint(model_id, {"run_id": run_id, "checkpoint": checkpoint_id})
        return _extract(model_id, path, {**record, "models": checkpoint.get("models"), "model": model_id})
    except ValueError as error:
        return jsonify({"error": str(error)}), 409
