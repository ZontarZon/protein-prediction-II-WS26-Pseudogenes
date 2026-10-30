from flask import Blueprint, current_app, jsonify, request, send_file

inference_bp = Blueprint("inference", __name__)


@inference_bp.get("/inference/models")
def inference_models():
    return jsonify({"models": current_app.config["MODEL_REGISTRY"].inference_models()})


@inference_bp.get("/inference/models/<model_id>/checkpoints")
def model_checkpoints(model_id):
    models = current_app.config["MODEL_REGISTRY"]
    if models.get(model_id) is None:
        return jsonify({"error": "Model not found"}), 404
    manager = current_app.config["RUN_MANAGER"]
    checkpoints = []
    for run in manager.list():
        config_models = (run.get("configuration") or {}).get("models", {})
        if model_id not in config_models.values():
            continue
        for checkpoint in manager.checkpoints(run["id"]):
            if checkpoint.get("path"):
                checkpoints.append({**checkpoint, "run_id": run["id"], "run_name": run.get("name"), "trainer": run.get("trainer")})
    for checkpoint in manager.committed_checkpoints(model_id):
        checkpoints.append({**checkpoint, "committed_id": checkpoint["id"], "run_name": checkpoint.get("run_name") or checkpoint["run_id"],
                            "id": checkpoint.get("checkpoint", checkpoint["id"]), "run_id": checkpoint.get("run_id")})
    weight_manager = current_app.config.get("WEIGHT_MANAGER")
    for weight in (weight_manager.list(model_id) if weight_manager else []):
        checkpoints.append({"id": weight["id"], "weight_id": weight["id"], "run_id": "", "run_name": weight.get("name", "Saved weights"),
                            "created_at": weight.get("created_at"), "epoch": None, "step": None, "loss": None})
    return jsonify({"checkpoints": checkpoints})


@inference_bp.post("/inference/runs/validate")
def validate_inference():
    config = request.get_json(silent=True) or {}
    errors = current_app.config["INFERENCE_RUN_MANAGER"].validate(config)
    return jsonify({"valid": not errors, "errors": errors, "configuration": config}), 200 if not errors else 400


@inference_bp.post("/inference/runs/<run_id>/rerun")
def rerun_inference(run_id):
    manager = current_app.config["INFERENCE_RUN_MANAGER"]
    config = request.get_json(silent=True) or {}
    try:
        return jsonify(manager.rerun(run_id, config)), 202
    except ValueError as error:
        return jsonify({"error": str(error)}), 404 if "not found" in str(error).lower() else 409


@inference_bp.delete("/inference/runs/<run_id>/outputs/<output_id>")
def delete_inference_output(run_id, output_id):
    try:
        result = current_app.config["INFERENCE_RUN_MANAGER"].delete_output(run_id, output_id)
        return (jsonify(result), 200) if result else (jsonify({"error": "Inference output not found"}), 404)
    except ValueError as error:
        return jsonify({"error": str(error)}), 409


@inference_bp.route("/inference/runs", methods=["GET", "POST"])
def inference_runs():
    manager = current_app.config["INFERENCE_RUN_MANAGER"]
    if request.method == "GET":
        return jsonify({"runs": manager.list()})
    config = request.get_json(silent=True) or {}
    try:
        return jsonify(manager.start(config)), 201
    except ValueError as error:
        return jsonify({"error": "Invalid inference configuration", "errors": str(error).split("; ")}), 400


@inference_bp.get("/inference/runs/<run_id>")
def inference_run(run_id):
    state = current_app.config["INFERENCE_RUN_MANAGER"].get(run_id)
    return (jsonify(state), 200) if state else (jsonify({"error": "Inference run not found"}), 404)


@inference_bp.delete("/inference/runs/<run_id>")
def delete_inference_run(run_id):
    manager = current_app.config["INFERENCE_RUN_MANAGER"]
    try:
        result = manager.delete(run_id)
        return (jsonify(result), 200) if result else (jsonify({"error": "Inference run not found"}), 404)
    except ValueError as error:
        return jsonify({"error": str(error)}), 409


@inference_bp.get("/inference/runs/<run_id>/progress")
def inference_progress(run_id):
    state = current_app.config["INFERENCE_RUN_MANAGER"].get(run_id)
    return (jsonify({"status": state["status"], "progress": state["progress"], "intermediate_available": state["intermediate_available"]}), 200) if state else (jsonify({"error": "Inference run not found"}), 404)


@inference_bp.get("/inference/runs/<run_id>/output")
def inference_output(run_id):
    manager = current_app.config["INFERENCE_RUN_MANAGER"]
    state = manager.get(run_id)
    if not state:
        return jsonify({"error": "Inference run not found"}), 404
    result = manager.output(run_id, request.args.get("output_id"))
    if not result:
        return jsonify({"error": "Inference output is not ready"}), 409
    if result.get("file"):
        output_id = result.get("id")
        suffix = f"?output_id={output_id}" if output_id else ""
        result = {**result, "url": f"/api/inference/runs/{run_id}/output/file{suffix}"}
    return jsonify(result)


@inference_bp.get("/inference/runs/<run_id>/output/file")
def inference_output_file(run_id):
    path = current_app.config["INFERENCE_RUN_MANAGER"].output_path(run_id, output_id=request.args.get("output_id"))
    return send_file(path, mimetype="image/png") if path else (jsonify({"error": "Output file not found"}), 404)


@inference_bp.get("/inference/runs/<run_id>/intermediate")
def inference_intermediate(run_id):
    path = current_app.config["INFERENCE_RUN_MANAGER"].output_path(run_id, intermediate=True)
    return send_file(path, mimetype="image/png", max_age=0) if path else (jsonify({"error": "No intermediate output is available"}), 404)


@inference_bp.post("/inference/runs/<run_id>/stop")
def stop_inference(run_id):
    result, error = current_app.config["INFERENCE_RUN_MANAGER"].stop(run_id)
    return (jsonify(result), 200) if not error else (jsonify({"error": error}), 409 if error != "Inference run not found" else 404)
