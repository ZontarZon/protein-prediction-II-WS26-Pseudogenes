from flask import Blueprint, jsonify
from .common import directory, files, read_json
from backend.device_manager import available_devices, default_device

discovery_bp = Blueprint("discovery", __name__)

@discovery_bp.get("/devices")
def devices():
    available = available_devices()
    return jsonify({"devices": available, "default_device": default_device(available)["id"]})

@discovery_bp.get("/dataloaders")
def dataloaders():
    base = directory("DATALOADERS_DIR", "DataLoaders")
    return jsonify({"dataloaders": [{"id": p.stem, "name": p.stem, "path": str(p)} for p in files(base) if p.is_file() and p.suffix == ".py"]})

@discovery_bp.get("/outputs")
def outputs():
    base = directory("OUTPUTS_DIR", "ModelOutputs")
    return jsonify({"outputs": [{"name": p.name, "path": str(p), "kind": "directory" if p.is_dir() else p.suffix.lstrip(".")} for p in files(base)]})
