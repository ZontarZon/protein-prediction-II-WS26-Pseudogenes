import io
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class InferenceRunManager:
    def __init__(self, training_root: Path, model_registry, training_manager, outputs_dir=None):
        self.training_root = training_root
        self.model_registry = model_registry
        self.training_manager = training_manager
        self.weight_manager = None
        self.runs_root = (Path(outputs_dir) if outputs_dir is not None else training_root / "ModelOutputs") / "inference_runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._stops = {}
        self._recover_interrupted()

    def _folder(self, run_id):
        folder = (self.runs_root / run_id).resolve()
        return folder if folder.parent == self.runs_root.resolve() else None

    def _read(self, run_id):
        folder = self._folder(run_id)
        try:
            return json.loads((folder / "inference.json").read_text(encoding="utf-8")) if folder and folder.is_dir() else None
        except (OSError, ValueError):
            return None

    def _write(self, run_id, value):
        folder = self._folder(run_id)
        if folder:
            temporary = folder / "inference.json.tmp"
            temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
            temporary.replace(folder / "inference.json")

    def _recover_interrupted(self):
        for folder in self.runs_root.iterdir():
            if not folder.is_dir():
                continue
            try:
                state = json.loads((folder / "inference.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if state.get("status") in {"QUEUED", "RUNNING", "STOPPING"}:
                state.update({"status": "FAILED", "finished_at": time.time(), "error": "Backend restarted before inference finished."})
                self._write(folder.name, state)

    def validate(self, config):
        errors = []
        model_id = config.get("model")
        registered = self.model_registry.get(model_id or "")
        if registered is None or not registered[0].has_inference():
            return ["Select a model with a registered inference implementation"]
        metadata = registered[0].metadata()
        supplied = config.get("inputs") or {}
        if not isinstance(supplied, dict):
            return ["Inference inputs must be an object"]
        allowed = {item["key"] for item in metadata["inference_parameters"]}
        for key in supplied.keys() - allowed:
            errors.append(f"Unknown input: {key}")
        for item in metadata["inference_parameters"]:
            value = supplied.get(item["key"], item.get("default"))
            if item.get("required") and value is None:
                errors.append(f"{item['label']} is required")
                continue
            if value is None:
                continue
            kind = item.get("type", "string")
            valid = (isinstance(value, int) and not isinstance(value, bool)) if kind == "integer" else (isinstance(value, (int, float)) and not isinstance(value, bool)) if kind == "float" else isinstance(value, bool) if kind == "boolean" else value in item.get("options", []) if kind == "selection" else isinstance(value, str)
            if not valid:
                errors.append(f"{item['label']} must be {kind}")
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not np.isfinite(value) or value < item.get("minimum", float("-inf")) or value > item.get("maximum", float("inf")):
                    errors.append(f"{item['label']} is outside its supported range")
        reference = config.get("checkpoint")
        if metadata.get("checkpoint_required") and not reference:
            errors.append("A trained checkpoint is required for this model")
        if reference:
            try:
                self.resolve_checkpoint(model_id, reference)
            except ValueError as error:
                errors.append(str(error))
        return errors

    def resolve_checkpoint(self, model_id, reference):
        if not isinstance(reference, dict):
            raise ValueError("Checkpoint reference must be an object")
        if isinstance(reference, dict) and reference.get("weight_id"):
            if not self.weight_manager:
                raise ValueError("Model weight storage is unavailable")
            record = self.weight_manager.get(reference["weight_id"])
            path = self.weight_manager.path(reference["weight_id"])
            if not record or not path:
                raise ValueError("Model weight artifact was not found")
            if record.get("model") != model_id:
                raise ValueError(f"Weight artifact is for {record.get('model')}, not {model_id}")
            return path
        if isinstance(reference, dict) and reference.get("committed_id"):
            committed = next((item for item in self.training_manager.committed_checkpoints(model_id)
                              if item.get("id") == reference["committed_id"]), None)
            if committed is None:
                raise ValueError("Committed checkpoint was not found for this model")
            artifact = (self.training_manager.outputs_root / committed.get("checkpoint_path", "")).resolve()
            if self.training_manager.outputs_root.resolve() not in artifact.parents or not artifact.is_file():
                raise ValueError("Committed checkpoint model weights are unavailable")
            return artifact
        if not isinstance(reference, dict) or not reference.get("run_id") or not reference.get("checkpoint"):
            raise ValueError("Checkpoint must identify a training run and checkpoint")
        run = self.training_manager.get(reference["run_id"])
        if run is None:
            raise ValueError("Checkpoint training run was not found")
        models = (run.get("configuration") or {}).get("models", {})
        checkpoint_info = next((item for item in self.training_manager.checkpoints(reference["run_id"])
                                if item["id"] == Path(str(reference["checkpoint"])).name), None)
        if model_id not in ((checkpoint_info or {}).get("models") or models).values():
            raise ValueError("Checkpoint was created from a different model")
        checkpoint = checkpoint_info
        if checkpoint is None:
            raise ValueError("Checkpoint was not found in the selected run")
        run_folder = self.training_manager._run_dir(reference["run_id"])
        artifact = (run_folder / "checkpoints" / f"{checkpoint['id']}.pt").resolve()
        if artifact.parent != (run_folder / "checkpoints").resolve() or not artifact.is_file():
            raise ValueError("Checkpoint model weights are unavailable")
        return artifact

    def start(self, config):
        errors = self.validate(config)
        if errors:
            raise ValueError("; ".join(errors))
        model_id = config["model"]
        model_type, _ = self.model_registry.get(model_id)
        run_id = uuid.uuid4().hex
        folder = self._folder(run_id)
        folder.mkdir(parents=True)
        state = {"id": run_id, "name": config.get("name") or f"Inference · {run_id[:8]}",
                 "model": model_id, "configuration": config, "output_type": model_type.metadata()["output"]["type"],
                 "status": "QUEUED", "created_at": time.time(), "started_at": None, "finished_at": None,
                 "progress": {"step": 0, "total_steps": None, "percent": 0.0, "elapsed_seconds": 0.0},
                 "outputs": []}
        self._write(run_id, state)
        stop_event = threading.Event()
        self._stops[run_id] = stop_event
        thread = threading.Thread(target=self._execute, args=(run_id, config, stop_event), name=f"inference-{run_id[:8]}", daemon=True)
        thread.start()
        return self.get(run_id)

    def rerun(self, run_id, config):
        errors = self.validate(config)
        if errors:
            raise ValueError("; ".join(errors))
        with self._lock:
            state = self._read(run_id)
            if state is None:
                raise ValueError("Inference run not found")
            if state.get("status") in {"QUEUED", "RUNNING", "STOPPING"}:
                raise ValueError("Wait for the current inference to finish before running it again")
            self._preserve_legacy_output(run_id, state)
            model_id = config["model"]
            model_type = self.model_registry.get(model_id)[0]
            state.update({"model": model_id, "configuration": config,
                          "output_type": model_type.metadata()["output"]["type"],
                          "status": "QUEUED", "started_at": None, "finished_at": None,
                          "progress": {"step": 0, "total_steps": None, "percent": 0.0, "elapsed_seconds": 0.0}})
            state.pop("error", None)
            state.pop("intermediate", None)
            (self._folder(run_id) / "intermediate.png").unlink(missing_ok=True)
            self._write(run_id, state)
            stop_event = threading.Event()
            self._stops[run_id] = stop_event
            thread = threading.Thread(target=self._execute, args=(run_id, config, stop_event),
                                      name=f"inference-{run_id[:8]}", daemon=True)
            thread.start()
        return self.get(run_id)

    def _preserve_legacy_output(self, run_id, state):
        """Move the original single-output format into the new output history."""
        outputs = list(state.get("outputs") or [])
        legacy = state.get("output")
        if not outputs and legacy:
            record = {**legacy, "id": "legacy", "created_at": state.get("finished_at"),
                      "configuration": state.get("configuration", {})}
            old_file = legacy.get("file")
            if old_file:
                source = (self._folder(run_id) / old_file).resolve()
                folder = self._folder(run_id).resolve()
                if folder not in source.parents or not source.is_file():
                    raise ValueError("The saved inference output is unavailable")
                extension = source.suffix or ".bin"
                relative = Path("outputs") / f"legacy{extension}"
                destination = (folder / relative).resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source != destination:
                    shutil.copy2(source, destination)
                record["file"] = str(relative)
            outputs.append(record)
        state["outputs"] = outputs

    def _save_intermediate(self, run_id, output, output_id=None):
        folder = self._folder(run_id)
        if isinstance(output, torch.Tensor):
            tensor = output.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
            if array.min(initial=0) < 0:
                array = (array.clip(-1, 1) + 1) * 127.5
            elif array.max(initial=0) <= 1:
                array = array * 255
            output = Image.fromarray(array.clip(0, 255).astype(np.uint8))
        if isinstance(output, np.ndarray):
            array = output
            if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))
            if np.issubdtype(array.dtype, np.floating):
                if array.size and np.isfinite(array).all() and array.min() >= -1 and array.max() <= 1:
                    array = (array.clip(-1, 1) + 1) * 127.5 if array.min() < 0 else array * 255
            output = Image.fromarray(array.clip(0, 255).astype(np.uint8))
        if isinstance(output, Image.Image):
            relative = Path("outputs") / f"{output_id}.png" if output_id else Path("intermediate.png")
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
            output.convert("RGB").save(temporary, format="PNG")
            temporary.replace(path)
            return {"type": "image", "file": str(relative)}
        return {"type": "json", "value": output}

    def create_model(self, config):
        """Resolve model settings and weights consistently for single and dataset inference."""
        model_parameters = {}
        reference = config.get("checkpoint") or {}
        if reference.get("weight_id") and self.weight_manager:
            artifact = self.weight_manager.get(reference["weight_id"])
            model_parameters = (artifact or {}).get("model_config", {})
        elif reference.get("committed_id"):
            record = next((item for item in self.training_manager.committed_checkpoints(config["model"]) if item.get("id") == reference["committed_id"]), {})
            roles = record.get("configuration", {}).get("models", {})
            role = next((key for key, value in roles.items() if value == config["model"]), None)
            model_parameters = record.get("model_configuration", {}).get(role, {}) if role else {}
        model = self.model_registry.create(config["model"], model_parameters)
        checkpoint = self.resolve_checkpoint(config["model"], reference) if reference else ""
        return model, checkpoint

    def _execute(self, run_id, config, stop_event):
        folder = self._folder(run_id)
        with self._lock:
            state = self._read(run_id) or {}
            state.update({"status": "RUNNING", "started_at": time.time()})
            self._write(run_id, state)
        started = time.monotonic()
        model_type = self.model_registry.get(config["model"])[0]
        try:
            model, checkpoint = self.create_model(config)

            def report(step, total_steps, output=None, **extra):
                if stop_event.is_set():
                    raise InterruptedError("Inference stopped by user")
                intermediate = self._save_intermediate(run_id, output) if output is not None else None
                with self._lock:
                    latest = self._read(run_id) or {}
                    progress = {"step": step, "total_steps": total_steps, "percent": step * 100.0 / max(total_steps, 1),
                                "elapsed_seconds": time.monotonic() - started, "timestamp": time.time(), **extra}
                    latest.update({"progress": progress, "intermediate": intermediate or latest.get("intermediate")})
                    self._write(run_id, latest)

            output = model.infer(config.get("inputs", {}), str(checkpoint), progress_callback=report)
            output_id = uuid.uuid4().hex
            result = self._save_intermediate(run_id, output, output_id) if model_type.metadata()["output"]["type"] == "image" else {"type": model_type.metadata()["output"]["type"], "value": output}
            latest = self._read(run_id) or {}
            record = {"id": output_id, **result, "created_at": time.time(), "configuration": config}
            with self._lock:
                latest = self._read(run_id) or latest
                latest.setdefault("outputs", []).append(record)
                latest.update({"status": "COMPLETED", "finished_at": time.time(), "output": record})
                self._write(run_id, latest)
        except InterruptedError:
            self._finish(run_id, "STOPPED")
        except Exception as error:
            self._finish(run_id, "FAILED", str(error))
        finally:
            with self._lock:
                if self._stops.get(run_id) is stop_event:
                    self._stops.pop(run_id, None)

    def _finish(self, run_id, status, error=None):
        with self._lock:
            state = self._read(run_id) or {}
            state.update({"status": status, "finished_at": time.time()})
            if error:
                state["error"] = error
            self._write(run_id, state)

    def get(self, run_id):
        state = self._read(run_id)
        if state is None:
            return None
        if not state.get("outputs") and state.get("output"):
            state["outputs"] = [{**state["output"], "id": state["output"].get("id", "legacy"),
                                 "created_at": state["output"].get("created_at", state.get("finished_at")),
                                 "configuration": state["output"].get("configuration", state.get("configuration", {}))}]
        state["intermediate_available"] = bool((self._folder(run_id) / "intermediate.png").is_file())
        reference = ((state.get("configuration") or {}).get("checkpoint") or {})
        if reference.get("committed_id"):
            state["checkpoint_reference_status"] = "available" if any(
                item.get("id") == reference["committed_id"]
                for item in self.training_manager.committed_checkpoints((state.get("configuration") or {}).get("model"))
            ) else "deleted"
        return state

    def list(self):
        items = []
        for folder in sorted(self.runs_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if folder.is_dir():
                state = self.get(folder.name)
                if state:
                    items.append(state)
        return items

    def output(self, run_id, output_id=None):
        state = self._read(run_id)
        if not state:
            return None
        records = state.get("outputs") or ([state["output"]] if state.get("output") else [])
        if output_id is None:
            return state.get("output") or (records[-1] if records else None)
        return next((item for item in records if item.get("id", "legacy") == output_id), None)

    def output_path(self, run_id, intermediate=False, output_id=None):
        state = self._read(run_id)
        if not state:
            return None
        record = self.output(run_id, output_id) if not intermediate else None
        file = "intermediate.png" if intermediate else (record or {}).get("file")
        folder = self._folder(run_id).resolve()
        path = (folder / file).resolve() if file else None
        return path if path and folder in path.parents and path.is_file() else None

    def delete_output(self, run_id, output_id):
        with self._lock:
            state = self._read(run_id)
            if state is None:
                return None
            records = list(state.get("outputs") or [])
            if not records and state.get("output"):
                self._preserve_legacy_output(run_id, state)
                records = list(state["outputs"])
            record = next((item for item in records if item.get("id", "legacy") == output_id), None)
            if record is None:
                return None
            file = record.get("file")
            path = None
            if file:
                folder = self._folder(run_id).resolve()
                path = (folder / file).resolve()
                if folder not in path.parents:
                    raise ValueError("Inference output path is invalid")
            remaining = [item for item in records if item.get("id", "legacy") != output_id]
            state["outputs"] = remaining
            if (state.get("output") or {}).get("id", "legacy") == output_id:
                if remaining:
                    state["output"] = remaining[-1]
                else:
                    state.pop("output", None)
            self._write(run_id, state)
            if path:
                path.unlink(missing_ok=True)
            return {"id": output_id, "deleted": True}

    def stop(self, run_id):
        state = self._read(run_id)
        if not state:
            return None, "Inference run not found"
        event = self._stops.get(run_id)
        if state.get("status") != "RUNNING" or event is None:
            return None, f"Cannot stop inference in {state.get('status')} state"
        event.set()
        state["status"] = "STOPPING"
        self._write(run_id, state)
        return self.get(run_id), None

    def delete(self, run_id):
        with self._lock:
            state = self._read(run_id)
            if state is None:
                return None
            if state.get("status") in {"QUEUED", "RUNNING", "STOPPING"}:
                raise ValueError("Stop the inference and wait for it to finish before deleting it")
            folder = self._folder(run_id)
            if folder is None or not folder.is_dir():
                return None
            shutil.rmtree(folder)
            return {"id": run_id, "deleted": True}
