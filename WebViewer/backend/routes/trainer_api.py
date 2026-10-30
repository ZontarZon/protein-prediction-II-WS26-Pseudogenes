import math
from flask import Blueprint, current_app, jsonify, request
from backend.routes.common import model_records, dataset_records
from backend.device_manager import available_devices, default_device

trainer_api_bp = Blueprint("trainer_api", __name__)


def _schema(trainer_id):
    cls = current_app.config["TRAINER_REGISTRY"].get(trainer_id)
    return cls.configuration_schema() if cls else None


@trainer_api_bp.get("/trainers")
def trainers():
    return jsonify({"trainers": current_app.config["TRAINER_REGISTRY"].list()})


@trainer_api_bp.get("/trainers/<trainer_id>")
def trainer_detail(trainer_id):
    schema = _schema(trainer_id)
    if schema is None:
        return jsonify({"error": "Trainer not found"}), 404
    return jsonify({"id": trainer_id, "name": trainer_id, "schema": schema})


@trainer_api_bp.get("/trainers/<trainer_id>/schema")
def trainer_schema(trainer_id):
    schema = _schema(trainer_id)
    return (jsonify({"id": trainer_id, **schema}), 200) if schema is not None else (jsonify({"error": "Trainer not found"}), 404)


@trainer_api_bp.get("/trainers/<trainer_id>/dataset-statistics")
def trainer_dataset_statistics(trainer_id):
    trainer = current_app.config["TRAINER_REGISTRY"].get(trainer_id)
    if trainer is None:
        return jsonify({"error": "Trainer not found"}), 404
    role = request.args.get("role", "")
    if role not in {field["key"] for field in trainer.configuration_schema()["datasets"]}:
        return jsonify({"error": "Unknown dataset role"}), 400
    path = dataset_records().get(request.args.get("dataset", ""))
    if path is None:
        return jsonify({"error": "Dataset not found"}), 404
    from backend.dataset_statistics import dataset_statistics
    return jsonify(dataset_statistics(trainer, path, role))


def validate_configuration(config):
    errors = []
    trainer_id = config.get("trainer")
    trainer = current_app.config["TRAINER_REGISTRY"].get(trainer_id or "")
    if trainer is None:
        return ["A valid trainer is required"]
    available = available_devices()
    requested_device = config.get("device", default_device(available)["id"])
    if not isinstance(requested_device, str) or requested_device not in {item["id"] for item in available}:
        errors.append("Select a computing device available to this backend")
    schema = trainer.configuration_schema()
    models = {item["id"]: item for item in model_records()}
    datasets = set(dataset_records())
    for group, available in (("models", models), ("datasets", datasets)):
        supplied = config.get(group) or {}
        if not isinstance(supplied, dict):
            errors.append(f"{group.title()} must be an object keyed by trainer role")
            supplied = {}
        declared = {item["key"] for item in schema[group]}
        for unexpected in supplied.keys() - declared:
            errors.append(f"Unknown {group[:-1]} role: {unexpected}")
        for item in schema[group]:
            value = supplied.get(item["key"])
            if item["required"] and not value:
                errors.append(f"{item['label']} is required")
            elif value and value not in available:
                errors.append(f"{item['label']} references an unavailable {group[:-1]}")
            elif value and item.get("choices") and value not in item["choices"]:
                errors.append(f"{item['label']} must use one of the models supported by this trainer")
            elif value and item.get("interface") and item["interface"] not in available[value].get("interfaces", []):
                errors.append(f"{item['label']} must implement the {item['interface']} model interface")
    params = config.get("hyperparameters") or {}
    if not isinstance(params, dict):
        errors.append("Hyperparameters must be an object")
        params = {}
    supported_parameters = {item["key"] for item in schema["hyperparameters"]}
    for unexpected in params.keys() - supported_parameters:
        errors.append(f"Unsupported hyperparameter: {unexpected}")
    for item in schema["hyperparameters"]:
        value = params.get(item["key"], item.get("default"))
        if value is None and item["required"]:
            errors.append(f"{item['label']} is required")
            continue
        if value is None:
            continue
        try:
            if item["type"] == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif item["type"] == "float":
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif item["type"] == "boolean":
                valid = isinstance(value, bool)
            elif item["type"] == "selection":
                valid = value in item.get("options", [])
            else:
                valid = isinstance(value, str)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"{item['label']} must be {item['type']}")
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < item.get("minimum", -math.inf) or value > item.get("maximum", math.inf):
                errors.append(f"{item['label']} is outside its supported range")
        if isinstance(value, float):
            if not math.isfinite(value):
                errors.append(f"{item['label']} must be finite")
    weights = current_app.config.get("WEIGHT_MANAGER")
    if weights:
        errors.extend(weights.validate_initialization(config.get("initialization"), config.get("models") or {}))
    return errors


@trainer_api_bp.post("/training/validate")
def validate_run():
    config = request.get_json(silent=True) or {}
    errors = validate_configuration(config)
    return jsonify({"valid": not errors, "errors": errors, "configuration": config}), 200 if not errors else 400


@trainer_api_bp.route("/runs", methods=["GET", "POST"])
def runs():
    manager = current_app.config["RUN_MANAGER"]
    if request.method == "GET":
        return jsonify({"runs": manager.list()})
    config = request.get_json(silent=True) or {}
    errors = validate_configuration(config)
    if errors:
        return jsonify({"error": "Invalid training configuration", "errors": errors}), 400
    trainer_type = current_app.config["TRAINER_REGISTRY"].get(config["trainer"])
    config.setdefault("device", default_device()["id"])
    config = {**config, "hyperparameters": {**{item["key"]: item.get("default") for item in trainer_type.configuration_schema()["hyperparameters"] if "default" in item}, **(config.get("hyperparameters") or {})}}
    try:
        return jsonify(manager.create(config)), 201
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@trainer_api_bp.get("/runs/<run_id>")
def get_run(run_id):
    value = current_app.config["RUN_MANAGER"].get(run_id)
    return (jsonify(value), 200) if value else (jsonify({"error": "Run not found"}), 404)


@trainer_api_bp.get("/runs/<run_id>/metrics")
def run_metrics(run_id):
    values = current_app.config["RUN_MANAGER"].metrics(run_id)
    return (jsonify({"metrics": values}), 200) if values is not None else (jsonify({"error": "Run not found"}), 404)


@trainer_api_bp.get("/runs/<run_id>/terminal")
def run_terminal(run_id):
    try:
        offset = max(0, int(request.args.get("offset", "0")))
    except ValueError:
        return jsonify({"error": "Invalid terminal offset"}), 400
    value = current_app.config["RUN_MANAGER"].terminal_output(run_id, offset)
    return (jsonify(value), 200) if value is not None else (jsonify({"error": "Run not found"}), 404)


@trainer_api_bp.get("/runs/<run_id>/checkpoints")
def run_checkpoints(run_id):
    manager = current_app.config["RUN_MANAGER"]
    if manager.get(run_id) is None:
        return jsonify({"error": "Run not found"}), 404
    return jsonify({"checkpoints": manager.checkpoints(run_id)})


@trainer_api_bp.get("/runs/<run_id>/checkpoints/<checkpoint_id>")
def run_checkpoint(run_id, checkpoint_id):
    value = current_app.config["RUN_MANAGER"].checkpoint(run_id, checkpoint_id)
    if value:
        run = current_app.config["RUN_MANAGER"].get(run_id)
        value["run_id"] = run_id
        value["configuration"] = (run or {}).get("configuration", {})
    return (jsonify(value), 200) if value else (jsonify({"error": "Checkpoint not found"}), 404)


@trainer_api_bp.post("/runs/<run_id>/checkpoints/<checkpoint_id>/commit")
def commit_run_checkpoint(run_id, checkpoint_id):
    try:
        return jsonify(current_app.config["RUN_MANAGER"].commit_checkpoint(run_id, checkpoint_id)), 201
    except ValueError as error:
        return jsonify({"error": str(error)}), 404 if "not found" in str(error).lower() else 409


@trainer_api_bp.delete("/runs/<run_id>")
def delete_run(run_id):
    try:
        result = current_app.config["RUN_MANAGER"].delete(run_id)
        return (jsonify(result), 200) if result else (jsonify({"error": "Run not found"}), 404)
    except ValueError as error:
        return jsonify({"error": str(error)}), 409


@trainer_api_bp.delete("/runs")
def delete_all_runs():
    try:
        result = current_app.config["RUN_MANAGER"].delete_all()
        return jsonify(result), 200
    except ValueError as error:
        return jsonify({"error": str(error)}), 409


@trainer_api_bp.post("/runs/<run_id>/<action>")
def run_control(run_id, action):
    if action not in {"pause", "resume", "stop"}:
        return jsonify({"error": "Unknown run action"}), 404
    result, error = current_app.config["RUN_MANAGER"].control(run_id, action)
    return (jsonify(result), 200) if not error else (jsonify({"error": error}), 409 if error != "Run not found" else 404)
