import json
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

from backend.trainer_registry import TrainerRegistry
from backend.device_manager import resolve_device


TERMINAL = {"STOPPED", "COMPLETED", "FAILED"}


class _RunOutputStream:
    """Route writes from a training thread to its own append-only log."""
    def __init__(self, original, stream_name):
        self.original = original
        self.stream_name = stream_name
        self.targets = {}
        self.line_start = {}

    def write(self, value):
        if not isinstance(value, str):
            value = str(value)
        target = self.targets.get(threading.get_ident())
        if target is None:
            return self.original.write(value)
        if value:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            prefix = f"[{timestamp}] [{self.stream_name}] "
            lines = value.splitlines(keepends=True)
            if not lines:
                lines = [value]
            thread_id = threading.get_ident()
            at_line_start = self.line_start.get(thread_id, True)
            rendered_lines = []
            for line in lines:
                rendered_lines.append((prefix if at_line_start else "") + line)
                at_line_start = line.endswith(("\n", "\r"))
            self.line_start[thread_id] = at_line_start
            rendered = "".join(rendered_lines)
            with _log_write_lock:
                with target.open("a", encoding="utf-8", errors="replace") as log:
                    log.write(rendered)
                    log.flush()
        return len(value)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


_stdout_router = None
_stderr_router = None
_router_lock = threading.Lock()
_log_write_lock = threading.RLock()


def _install_output_routers():
    global _stdout_router, _stderr_router
    with _router_lock:
        if _stdout_router is None:
            _stdout_router = _RunOutputStream(sys.stdout, "OUT")
            _stderr_router = _RunOutputStream(sys.stderr, "ERR")
        # Some server runners and notebook/debug tooling replace the process
        # streams after application initialization. Reattach before a run so
        # ordinary print() calls still pass through the run-aware routers.
        if sys.stdout is not _stdout_router:
            _stdout_router.original = sys.stdout
            sys.stdout = _stdout_router
        if sys.stderr is not _stderr_router:
            _stderr_router.original = sys.stderr
            sys.stderr = _stderr_router
    return _stdout_router, _stderr_router


class TrainingRunManager:
    def __init__(self, training_root: Path, project_root: Path, registry: TrainerRegistry, model_registry=None,
                 metric_persist_interval_seconds=2.0, metric_persist_interval_steps=50,
                 recent_metric_limit=1000, outputs_dir=None, dataset_roots=None):
        self.training_root = training_root
        self.project_root = project_root
        self.registry = registry
        self.model_registry = model_registry
        self.weight_manager = None
        self.dataset_roots = dataset_roots
        self.outputs_root = Path(outputs_dir) if outputs_dir is not None else training_root / "ModelOutputs"
        self.runs_root = self.outputs_root / "training_runs"
        self.committed_root = self.outputs_root / "CommittedCheckpoints"
        self.committed_index = self.outputs_root / "committed_checkpoints.json"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._stdout_router, self._stderr_router = _install_output_routers()
        self._lock = threading.RLock()
        self._persist_lock = threading.Lock()
        self._trainers = {}
        self._threads = {}
        self._active_runs = {}
        self.metric_persist_interval_seconds = max(0.1, float(metric_persist_interval_seconds))
        self.metric_persist_interval_steps = max(1, int(metric_persist_interval_steps))
        self.recent_metric_limit = max(1, int(recent_metric_limit))
        self._persistence_wakeup = threading.Event()
        self._persistence_thread = threading.Thread(target=self._persistence_loop, name="training-run-persistence", daemon=True)
        self._recover_interrupted_runs()
        try:
            self._preserve_committed({item.get("run_id") for item in self.committed_checkpoints() if item.get("run_id")})
        except (OSError, ValueError):
            # Keep the service available; APIs report unresolved committed artifacts when accessed.
            pass
        self._persistence_thread.start()

    def _run_dir(self, run_id):
        candidate = (self.runs_root / run_id).resolve()
        return candidate if candidate.parent == self.runs_root.resolve() else None

    def _read(self, run_id):
        folder = self._run_dir(run_id)
        if not folder or not folder.is_dir():
            return None
        try:
            return json.loads((folder / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, run_id, value):
        folder = self._run_dir(run_id)
        if folder is None:
            return
        temp = folder / "run.json.tmp"
        temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        temp.replace(folder / "run.json")

    def _recover_interrupted_runs(self):
        for folder in self.runs_root.iterdir():
            if not folder.is_dir():
                continue
            try:
                metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if metadata.get("status") in {"QUEUED", "RUNNING", "PAUSED", "STOPPING"}:
                metadata.update({"status": "FAILED", "finished_at": time.time(), "error": "Backend restarted before the run finished."})
                self._write(folder.name, metadata)

    def _metric_history(self, run_id):
        folder = self._run_dir(run_id)
        path = folder / "metrics.jsonl" if folder else None
        rows = []
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def _runtime_copy(self, run_id):
        with self._lock:
            state = self._active_runs.get(run_id)
            if state is None:
                return None
            result = dict(state["metadata"])
            result["current_metrics"] = dict(state["metadata"].get("current_metrics") or {})
            result["metrics"] = list(state["recent_metrics"])
            result["checkpoints"] = list(state["checkpoints"])
            result["recent_metrics_only"] = True
            return result

    def _update_runtime(self, run_id, updates, metric=None):
        wake_worker = bool(updates)
        with self._lock:
            state = self._active_runs.get(run_id)
            if state is None:
                return
            state["metadata"].update(updates)
            state["metadata"]["updated_at"] = time.time()
            state["dirty"] = True
            if metric is not None:
                metric = dict(metric)
                state["metadata"]["current_metrics"] = metric
                state["metadata"]["progress"] = metric.get("progress", state["metadata"].get("progress", 0.0))
                state["recent_metrics"].append(metric)
                state["pending_metrics"].append(metric)
                wake_worker = wake_worker or len(state["pending_metrics"]) >= self.metric_persist_interval_steps
        if wake_worker:
            self._persistence_wakeup.set()

    def _persist_run(self, run_id, force=False):
        """Persist a small state snapshot and any accumulated metric rows outside the state lock."""
        with self._persist_lock:
            now = time.monotonic()
            with self._lock:
                state = self._active_runs.get(run_id)
                if state is None:
                    return False
                due_by_steps = len(state["pending_metrics"]) >= self.metric_persist_interval_steps
                due_by_time = now - state["last_persisted"] >= self.metric_persist_interval_seconds
                if not force and not (due_by_steps or (due_by_time and state["dirty"])):
                    return False
                metadata = dict(state["metadata"])
                pending = list(state["pending_metrics"])
                state["pending_metrics"].clear()
                state["persisting_metrics"] = pending
                state["dirty"] = False
                state["last_persisted"] = now
            try:
                folder = self._run_dir(run_id)
                if pending and folder:
                    path = folder / "metrics.jsonl"
                    with path.open("a", encoding="utf-8") as stream:
                        stream.writelines(json.dumps(item, default=str) + "\n" for item in pending)
                self._write(run_id, metadata)
            except (OSError, TypeError, ValueError):
                with self._lock:
                    current = self._active_runs.get(run_id)
                    if current is not None:
                        current["pending_metrics"].extendleft(reversed(pending))
                        current["persisting_metrics"] = []
                        current["dirty"] = True
                return False
            with self._lock:
                current = self._active_runs.get(run_id)
                if current is not None:
                    current["persisting_metrics"] = []
            return metadata

    def _refresh_active_checkpoints(self, run_id):
        checkpoints = self.checkpoints(run_id)
        with self._lock:
            state = self._active_runs.get(run_id)
            if state is not None:
                state["checkpoints"] = checkpoints

    def _persistence_loop(self):
        while True:
            self._persistence_wakeup.wait(timeout=min(0.5, self.metric_persist_interval_seconds))
            self._persistence_wakeup.clear()
            now = time.monotonic()
            with self._lock:
                run_ids = list(self._active_runs)
                refresh = [run_id for run_id in run_ids
                           if now - self._active_runs[run_id]["last_checkpoint_refresh"] >= self.metric_persist_interval_seconds]
                for run_id in refresh:
                    self._active_runs[run_id]["last_checkpoint_refresh"] = now
            for run_id in run_ids:
                persisted = self._persist_run(run_id)
                if persisted and persisted.get("status") in TERMINAL:
                    try:
                        self._refresh_active_checkpoints(run_id)
                    except OSError:
                        pass
                    with self._lock:
                        self._active_runs.pop(run_id, None)
            for run_id in refresh:
                try:
                    self._refresh_active_checkpoints(run_id)
                except OSError:
                    pass

    def create(self, config):
        trainer_type = self.registry.get(config["trainer"])
        if trainer_type is None:
            raise ValueError("Unknown trainer")
        if config.get("resume_checkpoint"):
            self._resolve_resume_checkpoint(config["resume_checkpoint"])
        from backend.dataset_registry import dataset_catalog
        roots = self.dataset_roots if self.dataset_roots is not None else [self.training_root / "Datasets"]
        catalog = dataset_catalog(roots)
        selected = config.get("datasets", {})
        for dataset_id in selected.values():
            if dataset_id and dataset_id not in catalog:
                raise ValueError(f"Dataset is unavailable: {dataset_id}")
        config = {**config, "data_root": str(self.training_root),
                  "dataset_paths": {role: str(catalog[dataset_id]) for role, dataset_id in selected.items() if dataset_id}}

        run_id = uuid.uuid4().hex
        folder = self._run_dir(run_id)
        folder.mkdir(parents=True)
        clean_config = {key: value for key, value in config.items() if key not in {"data_root", "dataset_paths"}}
        metadata = {"id": run_id, "name": config.get("name") or f"{config['trainer']} · {run_id[:8]}",
                    "trainer": config["trainer"], "configuration": clean_config, "status": "QUEUED",
                    "created_at": time.time(), "started_at": None, "finished_at": None,
                    "current_metrics": {}, "progress": 0.0}
        self._write(run_id, metadata)
        with self._lock:
            self._active_runs[run_id] = {
                "metadata": metadata,
                "recent_metrics": deque(maxlen=self.recent_metric_limit),
                "pending_metrics": deque(),
                "persisting_metrics": [],
                "checkpoints": [],
                "last_persisted": time.monotonic(),
                "last_checkpoint_refresh": time.monotonic(),
                "dirty": False,
            }
        thread = threading.Thread(target=self._execute, args=(run_id, trainer_type, config), name=f"training-{run_id[:8]}", daemon=True)
        with self._lock:
            self._threads[run_id] = thread
        thread.start()
        return self.get(run_id)

    def _execute(self, run_id, trainer_type, config):
        self._stdout_router, self._stderr_router = _install_output_routers()
        folder = self._run_dir(run_id)
        thread_id = threading.get_ident()
        log_path = folder / "terminal.log"
        self._stdout_router.targets[thread_id] = log_path
        self._stderr_router.targets[thread_id] = log_path
        self._update_runtime(run_id, {"status": "RUNNING", "started_at": time.time()})
        self._persist_run(run_id, force=True)

        def report(metrics):
            metrics = {**metrics, "timestamp": metrics.get("timestamp", time.time())}
            self._update_runtime(run_id, {}, metric=metrics)

        trainer = None
        try:
            requested_device = config.get("device")
            selected_device, fallback = resolve_device(requested_device)
            config["device"] = selected_device["id"]
            trainer = trainer_type(config, folder, report)
            try:
                device_info = trainer.get_device_info() or selected_device
            except Exception:
                device_info = selected_device
            if isinstance(device_info, dict) and device_info.get("type"):
                device_info = {**selected_device, **device_info, "requested_id": requested_device or selected_device["id"]}
                actual_id = device_info.get("id", selected_device["id"])
                if fallback or actual_id != selected_device["id"]:
                    device_info["fallback"] = fallback or {"requested": selected_device["id"], "selected": actual_id,
                                                           "reason": "The trainer selected a different device than requested."}
                self._update_runtime(run_id, {"device": device_info})
                self._persist_run(run_id, force=True)
            resolved_models = {}
            for role, model_id in (config.get("models") or {}).items():
                if self.model_registry and self.model_registry.get(model_id):
                    resolved_models[role] = self.model_registry.create(model_id, (config.get("model_parameters") or {}).get(role, {}))
                    initialization = (config.get("initialization") or {}).get(role)
                    if initialization and self.weight_manager:
                        self.weight_manager.load_initialization(resolved_models[role], model_id, initialization, role)
                else:
                    resolved_models[role] = None
            trainer.bind_models(resolved_models)
            try:
                performance_info = trainer.get_performance_info()
            except Exception:
                performance_info = None
            if isinstance(performance_info, dict):
                self._update_runtime(run_id, {"performance": performance_info})
                self._persist_run(run_id, force=True)
            with self._lock:
                self._trainers[run_id] = trainer
            if config.get("resume_checkpoint"):
                reference = config["resume_checkpoint"]
                checkpoint = self._resolve_resume_checkpoint(reference)
                trainer.load_checkpoint(checkpoint)
                source = self.checkpoint(reference["run_id"], reference["checkpoint"])
                history = source.get("history", []) if source else []
                if history:
                    for item in history:
                        self._update_runtime(run_id, {}, metric=item)
            trainer.train()
            status = "STOPPED" if trainer.stop_requested else "COMPLETED"
            self._ensure_final_checkpoint(trainer, folder)
            self._finish(run_id, status)
        except Exception as error:
            traceback.print_exc()
            self._finish(run_id, "FAILED", str(error))
        finally:
            self._stdout_router.targets.pop(thread_id, None)
            self._stderr_router.targets.pop(thread_id, None)
            self._stdout_router.line_start.pop(thread_id, None)
            self._stderr_router.line_start.pop(thread_id, None)
            with self._lock:
                self._trainers.pop(run_id, None)

    def terminal_output(self, run_id, offset=0):
        folder = self._run_dir(run_id)
        if folder is None or self._read(run_id) is None:
            return None
        path = folder / "terminal.log"
        try:
            with path.open("rb") as log:
                start = min(max(0, int(offset)), path.stat().st_size)
                log.seek(start)
                raw = log.read(256 * 1024)
                content = ""
                while raw:
                    try:
                        content = raw.decode("utf-8")
                        break
                    except UnicodeDecodeError as error:
                        if error.reason != "unexpected end of data":
                            content = raw.decode("utf-8", errors="replace")
                            break
                        log.seek(error.start - len(raw), 1)
                        raw = raw[:error.start]
                        content = raw.decode("utf-8")
                next_offset = log.tell()
        except FileNotFoundError:
            return {"content": "", "offset": 0, "next_offset": 0}
        return {"content": content, "offset": start, "next_offset": next_offset}

    def _resolve_resume_checkpoint(self, reference):
        if not isinstance(reference, dict) or not reference.get("run_id") or not reference.get("checkpoint"):
            raise ValueError("resume_checkpoint must contain run_id and checkpoint")
        folder = self._run_dir(reference["run_id"])
        if folder is None:
            raise ValueError("Unknown source run")
        candidate = (folder / "checkpoints" / f"{reference['checkpoint']}.pt").resolve()
        if candidate.parent != (folder / "checkpoints").resolve() or not candidate.is_file():
            raise ValueError("Resume checkpoint was not found")
        return candidate

    def _ensure_final_checkpoint(self, trainer, folder):
        metrics = trainer.get_metrics()
        final_step = metrics.get("step")
        checkpoint_dir = folder / "checkpoints"
        if final_step is not None:
            for sidecar in checkpoint_dir.glob("*.json"):
                try:
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if metadata.get("step") == final_step and sidecar.with_suffix(".pt").is_file():
                    return sidecar.with_suffix(".pt")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        destination = checkpoint_dir / f"final_step_{int(final_step or 0):08d}.pt"
        trainer.save_checkpoint(destination)
        return destination

    def _finish(self, run_id, status, error=None):
        updates = {"status": status, "finished_at": time.time()}
        if error:
            updates["error"] = error
        with self._lock:
            state = self._active_runs.get(run_id)
            if state is None:
                metadata = self._read(run_id) or {}
                metadata.update(updates)
                self._write(run_id, metadata)
                return
            state["metadata"].update(updates)
            state["metadata"]["updated_at"] = time.time()
            state["dirty"] = True
        if self._persist_run(run_id, force=True):
            try:
                self._refresh_active_checkpoints(run_id)
            except OSError:
                pass
            with self._lock:
                self._active_runs.pop(run_id, None)

    def list(self):
        result = []
        with self._lock:
            active = {run_id: dict(state["metadata"]) for run_id, state in self._active_runs.items()}
        for folder in self.runs_root.iterdir():
            if folder.is_dir():
                metadata = active.get(folder.name) or self._read(folder.name)
                if metadata:
                    result.append(metadata)
        return sorted(result, key=lambda item: item.get("created_at", 0), reverse=True)

    def get(self, run_id):
        runtime = self._runtime_copy(run_id)
        if runtime is not None:
            return runtime
        value = self._read(run_id)
        if value is None:
            return None
        value["metrics"] = self._metric_history(run_id)
        value["checkpoints"] = self.checkpoints(run_id)
        return value

    def metrics(self, run_id):
        if self._runtime_copy(run_id) is not None:
            with self._persist_lock:
                history = self._metric_history(run_id)
                with self._lock:
                    state = self._active_runs.get(run_id)
                    pending = list(state["pending_metrics"]) if state else []
                    persisting = list(state["persisting_metrics"]) if state else []
                return history + persisting + pending
        return self._metric_history(run_id) if self._read(run_id) is not None else None

    def checkpoints(self, run_id):
        folder = self._run_dir(run_id)
        base = folder / "checkpoints" if folder else None
        if not base or not base.is_dir():
            return []
        checkpoints = []
        committed_by_checkpoint = {item["checkpoint"]: item for item in self.committed_checkpoints() if item["run_id"] == run_id}
        for sidecar in sorted(base.rglob("*.json")):
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
            artifact = sidecar.with_suffix(".pt")
            checkpoints.append({"id": sidecar.stem, "path": str(artifact.relative_to(folder)) if artifact.exists() else None,
                                "created_at": metadata.get("created_at"), "epoch": metadata.get("epoch"),
                                "step": metadata.get("step"), "loss": metadata.get("loss"), "metrics": metadata,
                                "trainer": metadata.get("trainer"), "models": metadata.get("models"),
                                "datasets": metadata.get("datasets"), "hyperparameters": metadata.get("hyperparameters"),
                                "committed": sidecar.stem in committed_by_checkpoint,
                                "committed_id": committed_by_checkpoint.get(sidecar.stem, {}).get("id")})
        return checkpoints

    def committed_checkpoints(self, model_id=None):
        try:
            records = json.loads(self.committed_index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            records = []
        if not isinstance(records, list):
            return []
        return [item for item in records if isinstance(item, dict) and (model_id is None or item.get("model") == model_id)]

    def delete_committed_checkpoint(self, committed_id):
        with self._lock:
            records = self.committed_checkpoints()
            record = next((item for item in records if item.get("id") == committed_id), None)
            if record is None:
                return None
            storage_id = str(record.get("storage_id") or record["id"])
            if Path(storage_id).name != storage_id:
                raise ValueError("Committed checkpoint storage metadata is invalid")
            folder = (self.committed_root / storage_id).resolve()
            if folder.parent != self.committed_root.resolve() or not folder.is_dir():
                raise ValueError("Committed checkpoint data is unavailable")
            artifact = (self.outputs_root / str(record.get("checkpoint_path", ""))).resolve()
            if artifact.parent != folder or artifact.name != "checkpoint.pt" or not artifact.is_file():
                raise ValueError("Committed checkpoint artifact is unavailable")

            remaining = [item for item in records if item.get("id") != committed_id]
            tombstone = self.committed_root / f".deleting-{storage_id}-{uuid.uuid4().hex}"
            folder.rename(tombstone)
            try:
                self._write_committed(remaining)
            except OSError:
                tombstone.rename(folder)
                raise
            shutil.rmtree(tombstone)
            return {"id": committed_id, "deleted": True}

    def _write_committed(self, records):
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        temporary = self.committed_index.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.committed_index)

    def commit_checkpoint(self, run_id, checkpoint_id):
        with self._lock:
            run = self._read(run_id)
            if run is None:
                raise ValueError("Training run not found")
            checkpoint = next((item for item in self.checkpoints(run_id) if item["id"] == Path(checkpoint_id).name), None)
            if checkpoint is None or not checkpoint.get("path"):
                raise ValueError("Checkpoint artifact not found")
            models = checkpoint.get("models") or (run.get("configuration") or {}).get("models") or {}
            if not models:
                raise ValueError("Checkpoint has no associated model")
            model_id = next(iter(models.values()))
            artifact = (self._run_dir(run_id) / checkpoint["path"]).resolve()
            if (self._run_dir(run_id) / "checkpoints").resolve() not in artifact.parents or not artifact.is_file():
                raise ValueError("Checkpoint artifact is unavailable")
            records = self.committed_checkpoints()
            committed_id = uuid.uuid4().hex
            record = {"id": committed_id, "model": model_id, "name": checkpoint["id"],
                      "run_id": run_id, "run_name": run.get("name"), "checkpoint": checkpoint["id"],
                      "created_at": time.time(), "source_created_at": checkpoint.get("created_at"),
                      "epoch": checkpoint.get("epoch"), "step": checkpoint.get("step"), "loss": checkpoint.get("loss"),
                      "trainer": checkpoint.get("trainer"), "datasets": checkpoint.get("datasets"),
                      "hyperparameters": checkpoint.get("hyperparameters"), "metrics": checkpoint.get("metrics"),
                      "history": checkpoint.get("history", []),
                      "configuration": run.get("configuration", {}),
                      "model_configuration": (run.get("configuration", {}).get("model_parameters", {}) or {}),
                      "model": model_id,
                      "checkpoint_format": artifact.suffix.lstrip("."),
                      "checkpoint_path": str((self.committed_root / committed_id / "checkpoint.pt").relative_to(self.outputs_root)),
                      "materialized": True}
            # Make the committed artifact independent immediately; run deletion is no longer part of its lifecycle.
            record_dir = self.outputs_root / Path(record["checkpoint_path"]).parent
            record_dir.mkdir(parents=True, exist_ok=True)
            destination = self.outputs_root / record["checkpoint_path"]
            temporary = destination.with_suffix(".pt.tmp")
            shutil.copy2(artifact, temporary)
            temporary.replace(destination)
            record["storage_id"] = record_dir.name
            self._write_committed_metadata(record)
            records.append(record)
            self._write_committed(records)
            return record

    def _write_committed_metadata(self, record):
        folder = self.outputs_root / Path(record["checkpoint_path"]).parent
        metadata_path = folder / "metadata.json"
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        temporary.replace(metadata_path)

    def _preserve_committed(self, run_ids):
        ids = set(run_ids)
        records = self.committed_checkpoints()
        updated = False
        for item in records:
            if item.get("run_id") not in ids or item.get("materialized"):
                continue
            current = (self.outputs_root / item.get("checkpoint_path", "")).resolve()
            if self.outputs_root.resolve() not in current.parents or (item.get("materialized") and current.is_file()):
                continue
            source_run = self._run_dir(item["run_id"])
            source = (source_run / "checkpoints" / f"{item['checkpoint']}.pt").resolve() if source_run else None
            if source is None or source_run.resolve() not in source.parents or not source.is_file():
                raise ValueError(f"Committed checkpoint {item['id']} is unavailable; refusing to delete its training run")
            details = self.checkpoint(item["run_id"], item["checkpoint"])
            run_record = self._read(item["run_id"]) or {}
            if details:
                for key in ("epoch", "step", "loss", "trainer", "datasets", "hyperparameters", "metrics", "history"):
                    if details.get(key) is not None:
                        item[key] = details[key]
                item["configuration"] = run_record.get("configuration", {})
                item["model_configuration"] = item["configuration"].get("model_parameters", {})
            storage_id = item.get("storage_id") or item["id"]
            destination_dir = self.committed_root / storage_id
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / "checkpoint.pt"
            temporary = destination.with_suffix(".pt.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            item["checkpoint_path"] = str(destination.relative_to(self.outputs_root))
            item["storage_id"] = storage_id
            item["materialized"] = True
            self._write_committed_metadata(item)
            updated = True
        if updated:
            self._write_committed(records)

    def delete(self, run_id):
        with self._lock:
            run = self._read(run_id)
            folder = self._run_dir(run_id)
            if run is None or folder is None:
                return None
            if run.get("status") not in TERMINAL:
                raise ValueError(f"Cannot delete a run in {run.get('status')} state")
            self._preserve_committed([run_id])
            shutil.rmtree(folder)
            return {"id": run_id, "deleted": True}

    def delete_all(self):
        with self._lock:
            runs = self.list()
            active = [item for item in runs if item.get("status") not in TERMINAL]
            if active:
                raise ValueError(f"Stop all active runs before deleting runs ({len(active)} still active)")
            ids = [item["id"] for item in runs]
            self._preserve_committed(ids)
            for run_id in ids:
                folder = self._run_dir(run_id)
                if folder and folder.exists():
                    shutil.rmtree(folder)
            return {"deleted": len(ids)}

    def checkpoint(self, run_id, checkpoint_id):
        safe_id = Path(checkpoint_id).name
        found = next((item for item in self.checkpoints(run_id) if item["id"] == safe_id), None)
        if found is None:
            return None
        history = self.metrics(run_id) or []
        checkpoint_step = found.get("step", float("inf"))
        found["history"] = [row for row in history if row.get("step", 0) <= checkpoint_step]
        return found

    def control(self, run_id, action):
        with self._lock:
            state = self._active_runs.get(run_id)
            metadata = state["metadata"] if state else self._read(run_id)
            trainer = self._trainers.get(run_id)
            if metadata is None:
                return None, "Run not found"
            status = metadata.get("status")
            if action == "pause" and status == "RUNNING" and trainer:
                trainer.pause()
                metadata["status"] = "PAUSED"
            elif action == "resume" and status == "PAUSED" and trainer:
                trainer.resume()
                metadata["status"] = "RUNNING"
            elif action == "stop" and status in {"RUNNING", "PAUSED"} and trainer:
                trainer.stop()
                metadata["status"] = "STOPPING"
            else:
                return None, f"Cannot {action} a run in {status} state"
            metadata["updated_at"] = time.time()
            if state:
                state["dirty"] = True
            else:
                self._write(run_id, metadata)
        if state:
            self._persist_run(run_id, force=True)
        return self.get(run_id), None
