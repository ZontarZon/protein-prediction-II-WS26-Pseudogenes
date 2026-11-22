import json
from copy import deepcopy
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__:
    from .Classefyer import GELUClassefyer
    from .DataLoader import load_splits
    from .HyperparameterTuner import tune
else:
    from Classefyer import GELUClassefyer
    from DataLoader import load_splits
    from HyperparameterTuner import tune

hparams = {
    "dataset_path": Path(__file__).resolve().parents[1] / "dataset" / "clusters.json",
    "checkpoint_dir": Path(__file__).resolve().parent / "Checkpoints",
    "metrics_dir": Path(__file__).resolve().parent / "Metrics",
    "seed": 42,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "hidden_sizes": (512, 64),
    "batch_size": 16,
    "epochs": 256,
    "early_stopping_patience": 256,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "min_learning_rate": 1e-8,
}


TUNING_EPOCHS = 20
TUNING_TRIALS = 50
SEARCH_RANGES = {
    "learning_rate": (1e-4, 1e-3),
    "weight_decay": (0.001, 0.1),
    "early_stopping_patience": (1, 200),
    "batch_size": (2, 32),
}


@torch.inference_mode()
def evaluate(model, batches, criterion, device):
    model.eval()
    loss_sum = correct = 0
    for embeddings, labels in batches:
        embeddings, labels = embeddings.to(device), labels.to(device)
        logits = model(embeddings)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return loss_sum / len(batches.dataset), correct / len(batches.dataset)


def save_metrics(path, metrics):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fit(settings, train_set, val_set, device, on_epoch=None):
    torch.manual_seed(settings["seed"])
    batch_size = settings["batch_size"]
    if len(train_set) < 2 or batch_size < 2:
        raise ValueError("Batch normalization requires at least two training samples per batch.")
    train_batches = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=len(train_set) % batch_size == 1
    )
    val_batches = DataLoader(val_set, batch_size=batch_size)
    model = GELUClassefyer(train_set.tensors[0].shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=settings["betas"],
        eps=settings["eps"],
    )
    epochs = settings["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=settings["min_learning_rate"]
    )

    best_loss = float("inf")
    best_weights = None
    best_epoch = None
    patience = settings["early_stopping_patience"]
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = train_correct = 0
        samples_seen = 0
        for embeddings, labels in tqdm(
            train_batches, desc=f"Epoch {epoch:02d}/{epochs}", unit="batch", disable=on_epoch is None
        ):
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            samples_seen += labels.size(0)
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.detach().argmax(dim=1) == labels).sum().item()

        val_loss, val_accuracy = evaluate(model, val_batches, criterion, device)
        if isfinite(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            if on_epoch is not None:
                best_weights = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        learning_rate = scheduler.get_last_lr()[0]
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / samples_seen,
            "train_accuracy": train_correct / samples_seen,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        }
        scheduler.step()
        stopped_early = 0 < patience <= epochs_without_improvement and epoch < epochs
        if on_epoch is not None:
            on_epoch(epoch_metrics, model, optimizer, scheduler, best_epoch, stopped_early)
        if stopped_early:
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, best_loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set = load_splits(
        hparams["dataset_path"], hparams["val_fraction"], hparams["test_fraction"], hparams["seed"]
    )

    def format_trial(settings):
        return ", ".join(f"{key}={settings[key]}" for key in SEARCH_RANGES)

    def score(settings):
        _, loss = fit({**settings, "epochs": TUNING_EPOCHS}, train_set, val_set, device)
        tqdm.write(f"Trial {format_trial(settings)} | best validation loss {loss:.4f}")
        return loss

    settings, trials = tune(hparams, SEARCH_RANGES, score, TUNING_TRIALS)
    tqdm.write(f"Selected {format_trial(settings)}")
    checkpoint_dir = Path(settings["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run-", dir=checkpoint_dir))
    metrics_dir = Path(settings["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_dir.name}.json"
    checkpoint_hparams = {
        key: str(value) if isinstance(value, Path) else value for key, value in settings.items()
    }
    metrics = {"hparams": checkpoint_hparams, "epochs": [], "best_epoch": None, "stopped_early": False}
    metrics["tuning"] = {"epochs": TUNING_EPOCHS, "trials": trials}
    tqdm.write(f"Checkpoint directory {run_dir}")
    tqdm.write(f"Metrics file {metrics_path}")

    def record_epoch(epoch_metrics, model, optimizer, scheduler, best_epoch, stopped_early):
        epoch = epoch_metrics["epoch"]
        torch.save(
            {
                **epoch_metrics,
                "embedding_dim": train_set.tensors[0].shape[1],
                "hparams": checkpoint_hparams,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            run_dir / f"epoch-{epoch:03d}.pt",
        )
        metrics["epochs"].append(epoch_metrics)
        metrics["best_epoch"] = best_epoch
        metrics["stopped_early"] = stopped_early
        save_metrics(metrics_path, metrics)
        tqdm.write(
            f"Epoch {epoch:02d} | validation loss {epoch_metrics['val_loss']:.4f} | "
            f"accuracy {epoch_metrics['val_accuracy']:.1%} | learning rate {epoch_metrics['learning_rate']:.2g}"
        )
        if stopped_early:
            tqdm.write(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")

    model, _ = fit(settings, train_set, val_set, device, record_epoch)
    if metrics["best_epoch"] is None:
        raise ValueError("Final training found no finite validation loss.")
    test_batches = DataLoader(test_set, batch_size=settings["batch_size"])
    test_loss, test_accuracy = evaluate(model, test_batches, nn.CrossEntropyLoss(), device)
    metrics["test"] = {"loss": test_loss, "accuracy": test_accuracy}
    save_metrics(metrics_path, metrics)
    print(f"Test loss {test_loss:.4f} | accuracy {test_accuracy:.1%}")
    return model


if __name__ == "__main__":
    train()
