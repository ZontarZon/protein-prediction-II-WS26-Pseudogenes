from pathlib import Path
import os
import sys
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.config import load_settings, configure_source_packages
from backend.dataset_registry import dataset_catalog
BACKEND_SETTINGS = load_settings()
BACKEND_PORT = BACKEND_SETTINGS["backend_port"]
configure_source_packages(BACKEND_SETTINGS)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
from routes.network_builder import network_builder_bp
from routes.models import models_bp
from routes.training import training_bp
from routes.datasets import datasets_bp
from routes.discovery import discovery_bp
from routes.trainer_api import trainer_api_bp
from backend.trainer_registry import TrainerRegistry
from backend.run_manager import TrainingRunManager
from backend.model_registry import ModelRegistry
from backend.inference_manager import InferenceRunManager
from backend.weight_manager import WeightManager
from routes.inference import inference_bp
from routes.weights import weights_bp
from routes.processing import processing_bp
from backend.processing_manager import ProcessingManager

ROOT = PROJECT_ROOT
DATA = BACKEND_SETTINGS["outputs_dir"].parent
app = Flask(__name__)
app.config["BACKEND_SETTINGS"] = BACKEND_SETTINGS
FRONTEND_STATIC_DIR = BACKEND_SETTINGS["frontend_static_dir"]
app.config["FRONTEND_STATIC_DIR"] = FRONTEND_STATIC_DIR
app.config["MAX_CONTENT_LENGTH"] = int(BACKEND_SETTINGS["max_request_size_mb"] * 1024 * 1024)
for key in ("models_dir", "trainers_dir", "dataloaders_dir", "dataset_viewers_dir", "outputs_dir", "legacy_checkpoints_dir", "dataset_roots"):
    app.config[key.upper()] = BACKEND_SETTINGS[key]
for key in ("models_dir", "trainers_dir", "dataloaders_dir"):
    if not BACKEND_SETTINGS[key].is_dir():
        raise RuntimeError(f"{key} does not exist or is not a directory: {BACKEND_SETTINGS[key]}")
# Fail early on missing storage mounts or ambiguous dataset IDs.
dataset_catalog(BACKEND_SETTINGS["dataset_roots"])
app.config["MODEL_TRAINING_ROOT"] = DATA
registry = TrainerRegistry(DATA, PROJECT_ROOT, trainers_dir=BACKEND_SETTINGS["trainers_dir"])
model_registry = ModelRegistry(DATA, PROJECT_ROOT, models_dir=BACKEND_SETTINGS["models_dir"])
app.config["TRAINER_REGISTRY"] = registry
app.config["MODEL_REGISTRY"] = model_registry
run_manager = TrainingRunManager(DATA, PROJECT_ROOT, registry, model_registry,
    outputs_dir=BACKEND_SETTINGS["outputs_dir"], dataset_roots=BACKEND_SETTINGS["dataset_roots"],
    metric_persist_interval_seconds=BACKEND_SETTINGS["metric_persist_interval_seconds"],
    metric_persist_interval_steps=BACKEND_SETTINGS["metric_persist_interval_steps"],
    recent_metric_limit=BACKEND_SETTINGS["recent_metric_limit"])
app.config["RUN_MANAGER"] = run_manager
app.config["INFERENCE_RUN_MANAGER"] = InferenceRunManager(DATA, model_registry, app.config["RUN_MANAGER"], outputs_dir=BACKEND_SETTINGS["outputs_dir"])
app.config["WEIGHT_MANAGER"] = WeightManager(DATA, model_registry, app.config["RUN_MANAGER"], outputs_dir=BACKEND_SETTINGS["outputs_dir"])
run_manager.weight_manager = app.config["WEIGHT_MANAGER"]
app.config["INFERENCE_RUN_MANAGER"].weight_manager = app.config["WEIGHT_MANAGER"]
app.config["PROCESSING_MANAGER"] = ProcessingManager(app.config["INFERENCE_RUN_MANAGER"], BACKEND_SETTINGS["dataset_roots"], BACKEND_SETTINGS["outputs_dir"])
CORS(app, resources={r"/api/*": {"origins": BACKEND_SETTINGS["cors_origins"]}})
for blueprint in (network_builder_bp, models_bp, training_bp, datasets_bp, discovery_bp, trainer_api_bp, inference_bp, weights_bp, processing_bp):
    app.register_blueprint(blueprint, url_prefix="/api")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/", defaults={"asset_path": ""})
@app.get("/<path:asset_path>")
def frontend(asset_path):
    """Serve the built SPA when frontend_static_dir is configured."""
    if asset_path == "api" or asset_path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    static_dir = app.config["FRONTEND_STATIC_DIR"]
    if static_dir is None:
        return jsonify({"error": "Frontend hosting is disabled; set frontend_static_dir in backend/settings.json"}), 404
    index = static_dir / "index.html"
    if not index.is_file():
        return jsonify({"error": f"Frontend build not found in {static_dir}"}), 404
    requested = static_dir / asset_path
    if asset_path and requested.is_file():
        return send_from_directory(static_dir, asset_path)
    if Path(asset_path).suffix:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(static_dir, "index.html")

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request exceeds max_request_size_mb in backend/settings.json"}), 413

@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    app.run(host=os.environ.get("DNATRAINER_BACKEND_HOST", BACKEND_SETTINGS["backend_host"]), port=BACKEND_PORT, debug=False)
