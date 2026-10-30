"""An actual, intentionally small CPU training example for the run system."""
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.trainers.base import TrainerBase, dataset, hyperparameter, model
from ModelsAndTrainers.Models.SampleNetworks import TinyDecoder, TinyEncoder


class DemoTrainer(TrainerBase):
    encoder = model("Encoder", "Sample encoder component supplied by the demo network module.", choices=["SampleNetworks"])
    decoder = model("Decoder", "Sample decoder component supplied by the demo network module.", choices=["SampleNetworks"])
    training_data = dataset("Training data", "Dataset used to create the sample reconstruction task.")
    learning_rate = hyperparameter("Learning rate", "float", 0.01)
    epochs = hyperparameter("Epochs", "integer", 8)
    batch_size = hyperparameter("Batch size", "integer", 16)
    checkpoint_interval = hyperparameter("Checkpoint interval", "integer", 2, description="Save a resumable checkpoint every N epochs.")
    cpu_threads = hyperparameter("CPU threads", "integer", 0, description="0 keeps PyTorch's current setting; a positive value sets intra-op threads for CPU training.")

    def __init__(self, config, run_dir, report):
        super().__init__(config, run_dir, report)
        self.device = torch.device(config.get("device", "cpu"))
        self.cpu_threads = max(0, int(config.get("hyperparameters", {}).get("cpu_threads", 0)))
        if self.device.type == "cpu" and self.cpu_threads:
            torch.set_num_threads(self.cpu_threads)
        torch.manual_seed(int(config.get("seed", 7)))
        if self.encoder != "SampleNetworks" or self.decoder != "SampleNetworks":
            raise ValueError("DemoTrainer requires the SampleNetworks asset for both model roles")
        self.encoder_net = TinyEncoder().to(self.device)
        self.decoder_net = TinyDecoder().to(self.device)
        self.optimizer = torch.optim.Adam(list(self.encoder_net.parameters()) + list(self.decoder_net.parameters()), lr=float(config["hyperparameters"].get("learning_rate", 0.01)))
        self.start_epoch = 0
        self.step = 0
        self.total_epochs = int(config["hyperparameters"].get("epochs", 8))
        self.learning_rate_value = float(config["hyperparameters"].get("learning_rate", 0.01))
        self.batch_size_value = int(config["hyperparameters"].get("batch_size", 16))
        self.checkpoint_interval = int(config["hyperparameters"].get("checkpoint_interval", 2))
        self.loss_function = nn.MSELoss()

    def get_device_info(self):
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(self.device)
        elif self.device.type == "xpu":
            name = torch.xpu.get_device_name(self.device)
        elif self.device.type == "mps":
            name = "Apple Metal (MPS)"
        else:
            name = "CPU"
        return {"type": self.device.type, "id": str(self.device), "name": name}

    def _load_samples(self):
        dataset_id = self.training_data
        dataset_path = self.dataset_path("training_data")
        vectors = []
        paths = sorted(dataset_path.glob("*.json"))[:64]
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            tags = record.get("tags", []) if isinstance(record, dict) else []
            try:
                short_id = int(record.get("short_id", 0) or 0) if isinstance(record, dict) else 0
            except (TypeError, ValueError):
                short_id = 0
            values = [len(tags) / 40.0, len(record) / 20.0 if isinstance(record, dict) else 0.0,
                      (sum(len(str(tag)) for tag in tags) % 100) / 100.0,
                      (short_id % 1000) / 1000.0]
            vectors.append(values)
        if not vectors:
            vectors = [[math.sin(i), math.cos(i), i / 16.0, (i % 3) / 3.0] for i in range(16)]
        return torch.tensor(vectors, dtype=torch.float32, device=self.device)

    def train(self):
        self._status = "RUNNING"
        samples = self._load_samples()
        loader = DataLoader(TensorDataset(samples), batch_size=self.batch_size_value, shuffle=True)
        started = time.monotonic()
        initial_step = self.step
        total_steps = max(1, (self.total_epochs - self.start_epoch) * len(loader))
        for epoch in range(self.start_epoch, self.total_epochs):
            if not self.check_control():
                break
            loss_total = torch.zeros((), device=self.device)
            batch_count = 0
            for (batch,) in loader:
                if not self.check_control():
                    break
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.decoder_net(self.encoder_net(batch))
                loss = self.loss_function(prediction, batch)
                loss.backward()
                self._sync_batch_samples = batch.shape[0]
                self.optimizer.step()
                self.step += 1
                loss_total += loss.detach()
                batch_count += 1
            if self.stop_requested:
                if batch_count:
                    elapsed = time.monotonic() - started
                    self.report({"epoch": epoch + 1, "step": self.step, "total_epochs": self.total_epochs,
                                 "total_steps": initial_step + total_steps,
                                 "progress": min(100.0, self.step * 100.0 / max(initial_step + total_steps, 1)),
                                 "loss": float((loss_total / batch_count).item()),
                                 "learning_rate": self.optimizer.param_groups[0]["lr"],
                                 "elapsed_seconds": elapsed, "timestamp": time.time()})
                break
            current_loss = float((loss_total / max(batch_count, 1)).item())
            elapsed = time.monotonic() - started
            completed_steps = self.step - initial_step
            remaining = (elapsed / completed_steps * max(total_steps - completed_steps, 0)) if completed_steps else None
            self.report({"epoch": epoch + 1, "step": self.step, "total_epochs": self.total_epochs,
                         "total_steps": initial_step + total_steps, "progress": min(100.0, (epoch + 1) * 100.0 / self.total_epochs),
                         "loss": current_loss, "learning_rate": self.optimizer.param_groups[0]["lr"],
                         "elapsed_seconds": elapsed, "estimated_remaining_seconds": remaining,
                         "timestamp": time.time()})
            self.start_epoch = epoch + 1
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(self.run_dir / "checkpoints" / f"checkpoint_{epoch + 1:04d}.pt")
        self.start_epoch = min(self.total_epochs, int(self.get_metrics().get("epoch", self.start_epoch)))
        self._status = "STOPPED" if self.stop_requested else "COMPLETED"

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.get_metrics(), "checkpoint": path.stem, "epoch": self.start_epoch, "step": self.step,
                    "created_at": time.time(), "trainer": type(self).__name__,
                    "models": self.config.get("models", {}), "datasets": self.config.get("datasets", {}),
                    "hyperparameters": self.config.get("hyperparameters", {})}
        torch.save({"encoder": self.encoder_net.state_dict(), "decoder": self.decoder_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epoch": self.start_epoch, "step": self.step,
                    "config": self.config, "metrics": self.get_metrics()}, path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_net.load_state_dict(state["encoder"])
        self.decoder_net.load_state_dict(state["decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.start_epoch = int(state.get("epoch", 0))
        self.step = int(state.get("step", 0))
        for key, value in state.get("metrics", {}).items():
            self._metrics[key] = value
