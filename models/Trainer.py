import json
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__:
    from .Classefyer import Classefyer, GELUClassefyer
    from .DataLoader import load_splits
else:
    from Classefyer import Classefyer, GELUClassefyer
    from DataLoader import load_splits

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


def train():
    torch.manual_seed(hparams["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set = load_splits(
        hparams["dataset_path"], hparams["val_fraction"], hparams["test_fraction"], hparams["seed"]
    )
    batch_size = hparams["batch_size"]
    train_batches = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_batches = DataLoader(val_set, batch_size=batch_size)
    test_batches = DataLoader(test_set, batch_size=batch_size)
    embedding_dim = train_set.tensors[0].shape[1]
    #model = Classefyer(embedding_dim, hparams["hidden_sizes"]).to(device)
    model = GELUClassefyer(embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        betas=hparams["betas"],
        eps=hparams["eps"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams["epochs"], eta_min=hparams["min_learning_rate"]
    )
    checkpoint_dir = Path(hparams["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run-", dir=checkpoint_dir))
    metrics_dir = Path(hparams["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_dir.name}.json"
    checkpoint_hparams = {
        key: str(value) if isinstance(value, Path) else value for key, value in hparams.items()
    }
    metrics = {"hparams": checkpoint_hparams, "epochs": [], "best_epoch": None, "stopped_early": False}
    tqdm.write(f"Checkpoint directory {run_dir}")
    tqdm.write(f"Metrics file {metrics_path}")

    best_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    patience = hparams["early_stopping_patience"]
    epochs_without_improvement = 0
    for epoch in range(1, hparams["epochs"] + 1):
        model.train()
        train_loss_sum = train_correct = 0
        for embeddings, labels in tqdm(train_batches, desc=f"Epoch {epoch:02d}/{hparams['epochs']}", unit="batch"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.detach().argmax(dim=1) == labels).sum().item()

        val_loss, val_accuracy = evaluate(model, val_batches, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            metrics["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        learning_rate = scheduler.get_last_lr()[0]
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / len(train_set),
            "train_accuracy": train_correct / len(train_set),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        }
        scheduler.step()
        torch.save(
            {
                **epoch_metrics,
                "embedding_dim": embedding_dim,
                "hparams": checkpoint_hparams,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            run_dir / f"epoch-{epoch:03d}.pt",
        )
        metrics["epochs"].append(epoch_metrics)
        metrics["stopped_early"] = (
            0 < patience <= epochs_without_improvement and epoch < hparams["epochs"]
        )
        save_metrics(metrics_path, metrics)
        tqdm.write(
            f"Epoch {epoch:02d} | validation loss {val_loss:.4f} | "
            f"accuracy {val_accuracy:.1%} | learning rate {learning_rate:.2g}"
        )
        if metrics["stopped_early"]:
            tqdm.write(f"Early stopping at epoch {epoch}. Best epoch was {metrics['best_epoch']}.")
            break

    model.load_state_dict(best_weights)
    test_loss, test_accuracy = evaluate(model, test_batches, criterion, device)
    metrics["test"] = {"loss": test_loss, "accuracy": test_accuracy}
    save_metrics(metrics_path, metrics)
    print(f"Test loss {test_loss:.4f} | accuracy {test_accuracy:.1%}")
    return model


if __name__ == "__main__":
    train()
