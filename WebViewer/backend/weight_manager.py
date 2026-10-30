"""Persistent model-only weights and generic model compatibility checks."""
import json
import shutil
import time
import uuid
from pathlib import Path

import torch


class WeightManager:
    def __init__(self, training_root, model_registry, run_manager, outputs_dir=None):
        self.outputs_root = Path(outputs_dir) if outputs_dir is not None else Path(training_root) / "ModelOutputs"
        self.weights_root = self.outputs_root / "ModelWeights"
        self.weights_root.mkdir(parents=True, exist_ok=True)
        self.model_registry = model_registry
        self.run_manager = run_manager

    def list(self, model_id=None):
        records = []
        for metadata_path in sorted(self.weights_root.glob("*/metadata.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (model_id is None or record.get("model") == model_id) and (metadata_path.parent / "weights.pt").is_file():
                records.append(record)
        return records

    def get(self, weight_id):
        for item in self.list():
            if item.get("id") == weight_id:
                return item
        return None

    def path(self, weight_id):
        record = self.get(weight_id)
        if not record:
            return None
        folder = (self.weights_root / weight_id).resolve()
        return folder / "weights.pt" if folder.parent == self.weights_root.resolve() and (folder / "weights.pt").is_file() else None

    def _new_artifact(self, model_id, weights, metadata):
        artifact_id = uuid.uuid4().hex
        folder = self.weights_root / artifact_id
        folder.mkdir(parents=True)
        torch.save({key: value.detach().cpu().contiguous() for key, value in weights.items()}, folder / "weights.pt")
        record = {"id": artifact_id, "model": model_id, "model_config": metadata.get("model_config", {}),
                  "source_checkpoint": metadata.get("source_checkpoint"), "source_training_run": metadata.get("source_training_run"),
                  "created_at": time.time(), "format": "pytorch_state_dict", "file": "weights.pt",
                  "path": str((folder / "weights.pt").relative_to(self.outputs_root)),
                  "inputs": metadata.get("inputs", []), "output": metadata.get("output", {}),
                  "trainer": metadata.get("trainer"), "dataset": metadata.get("dataset"),
                  "configuration": metadata.get("configuration", {}), "name": metadata.get("name") or f"{model_id} weights {artifact_id[:8]}",
                  "origin": metadata.get("origin", "extracted")}
        temporary = folder / "metadata.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        temporary.replace(folder / "metadata.json")
        return record

    def _load_payload(self, path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(f"Could not safely read this PyTorch weight artifact: {error}") from error

    def _weight_state(self, model_type, model_id, payload):
        model = self.model_registry.create(model_id)
        state = model.extract_weight_state(payload)
        try:
            model.load_weight_state(state, strict=True)
        except RuntimeError as error:
            raise ValueError(f"Weights are incompatible with {model_type.__name__}: {error}") from error
        return state

    def extract_from_checkpoint(self, model_id, checkpoint_path, source, model_config=None):
        registered = self.model_registry.get(model_id)
        if registered is None:
            raise ValueError(f"Model {model_id} does not provide a registered weight extraction interface")
        payload = self._load_payload(checkpoint_path)
        state = self._weight_state(registered[0], model_id, payload)
        role = next((key for key, value in (source.get("models") or source.get("configuration", {}).get("models", {})).items() if value == model_id), None)
        configured_model = source.get("model_configuration", {}).get(role, {}) if role else {}
        if not configured_model and role:
            configured_model = source.get("configuration", {}).get("model_parameters", {}).get(role, {})
        metadata = {"source_checkpoint": source.get("checkpoint"), "source_training_run": source.get("run_id"),
                    "model_config": model_config or configured_model, "inputs": registered[0].metadata().get("inputs"),
                    "output": registered[0].metadata().get("output"), "trainer": source.get("trainer"),
                    "dataset": source.get("datasets"), "configuration": source.get("configuration", {}),
                    "name": source.get("name"), "origin": "extracted"}
        return self._new_artifact(model_id, state, metadata)

    def import_file(self, file_storage, model_id, model_config=None):
        if not file_storage or not file_storage.filename:
            raise ValueError("Choose a PyTorch .pt or .pth weight file")
        if Path(file_storage.filename).suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("Only safe PyTorch .pt and .pth tensor state dictionaries are supported")
        registered = self.model_registry.get(model_id)
        if registered is None:
            raise ValueError("Select a registered model before importing weights")
        temporary = self.weights_root / f"upload-{uuid.uuid4().hex}.tmp"
        try:
            file_storage.save(temporary)
            payload = self._load_payload(temporary)
            candidate_model = self.model_registry.create(model_id, model_config or {})
            if isinstance(payload, dict) and "weights" in payload and isinstance(payload["weights"], dict):
                state = payload["weights"]
            elif isinstance(payload, dict) and any(key in payload and isinstance(payload[key], dict) for key in ("state_dict", "model_state_dict")):
                state = payload.get("state_dict", payload.get("model_state_dict"))
            elif isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
                state = payload
            else:
                raise ValueError("Imported file must contain a tensor state dictionary")
            try:
                candidate_model.load_weight_state(state, strict=True)
            except RuntimeError as error:
                raise ValueError(f"Imported weights are incompatible with {model_id}: {error}") from error
            safe_state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
            return self._new_artifact(model_id, safe_state, {"model_config": model_config or {}, "inputs": registered[0].metadata().get("inputs"),
                                      "output": registered[0].metadata().get("output"), "name": Path(file_storage.filename).name, "origin": "imported"})
        finally:
            temporary.unlink(missing_ok=True)

    def resolve_checkpoint(self, model_id, reference):
        if not isinstance(reference, dict):
            raise ValueError("Checkpoint reference must be an object")
        if reference.get("committed_id"):
            record = next((item for item in self.run_manager.committed_checkpoints(model_id) if item.get("id") == reference["committed_id"]), None)
            if not record:
                raise ValueError("Committed checkpoint was not found for this model")
            path = (self.run_manager.outputs_root / record.get("checkpoint_path", "")).resolve()
            if self.run_manager.outputs_root.resolve() not in path.parents or not path.is_file():
                raise ValueError("Committed checkpoint data is unavailable")
            return record, path
        run_id, checkpoint_id = reference.get("run_id"), reference.get("checkpoint")
        checkpoint = self.run_manager.checkpoint(run_id, checkpoint_id) if run_id and checkpoint_id else None
        run = self.run_manager.get(run_id) if checkpoint else None
        if not checkpoint or not run:
            raise ValueError("Source training checkpoint was not found")
        models = checkpoint.get("models") or run.get("configuration", {}).get("models", {})
        if model_id not in models.values():
            raise ValueError("Source checkpoint belongs to a different model")
        path = (self.run_manager._run_dir(run_id) / "checkpoints" / f"{checkpoint_id}.pt").resolve()
        if path.parent != (self.run_manager._run_dir(run_id) / "checkpoints").resolve() or not path.is_file():
            raise ValueError("Source checkpoint data is unavailable")
        return {**checkpoint, "run_id": run_id, "configuration": run.get("configuration", {})}, path

    def validate_initialization(self, initialization, models):
        errors = []
        if initialization is None:
            return errors
        if not isinstance(initialization, dict):
            return ["Initialization must be an object keyed by model role"]
        for role, config in initialization.items():
            if role not in (models or {}):
                errors.append(f"Initialization references unselected model role: {role}")
                continue
            if not isinstance(config, dict):
                errors.append(f"Initialization for {role} must be an object")
                continue
            model_id = models[role]
            if not isinstance(model_id, str):
                errors.append(f"Model for initialization role {role} must be a model identifier")
                continue
            registered = self.model_registry.get(model_id)
            if registered is None:
                errors.append(f"{model_id} does not support pretrained weight initialization")
                continue
            sources = [key for key in ("weight_id", "checkpoint", "committed_id") if config.get(key)]
            if len(sources) != 1:
                errors.append(f"Choose exactly one weight or checkpoint source for {role}")
                continue
            if not isinstance(config.get("freeze", False), bool):
                errors.append(f"Freeze setting for {role} must be boolean")
                continue
            if config.get("freeze") and not registered[0].supports_freezing:
                errors.append(f"{model_id} does not support frozen pretrained weights")
            try:
                if config.get("weight_id"):
                    artifact = self.get(config["weight_id"])
                    if not artifact:
                        raise ValueError("Selected weight artifact was not found")
                    if artifact.get("model") != model_id:
                        raise ValueError(f"Weight artifact is for {artifact.get('model')}, not {model_id}")
                    path = self.path(config["weight_id"])
                    state = self._load_payload(path)
                    registered_model = self.model_registry.create(model_id, artifact.get("model_config", {}))
                    registered_model.load_weight_state(state, strict=True)
                else:
                    reference = config.get("checkpoint") or {"committed_id": config.get("committed_id")}
                    source, path = self.resolve_checkpoint(model_id, reference)
                    state = self._load_payload(path)
                    model_parameters = source.get("model_configuration", {}).get(role, {})
                    if not model_parameters:
                        model_parameters = source.get("configuration", {}).get("model_parameters", {}).get(role, {})
                    registered_model = self.model_registry.create(model_id, model_parameters)
                    extracted = registered_model.extract_weight_state(state)
                    registered_model.load_weight_state(extracted, strict=True)
            except (ValueError, RuntimeError, TypeError) as error:
                errors.append(f"Initialization for {role} is incompatible: {error}")
        return errors

    def load_initialization(self, model, model_id, config, role):
        if config.get("weight_id"):
            record = self.get(config["weight_id"])
            state = self._load_payload(self.path(config["weight_id"]))
        else:
            reference = config.get("checkpoint") or {"committed_id": config.get("committed_id")}
            record, path = self.resolve_checkpoint(model_id, reference)
            state = model.extract_weight_state(self._load_payload(path))
        model.load_weight_state(state, strict=True)
        if config.get("freeze"):
            model.freeze_weights(True)
        return record
