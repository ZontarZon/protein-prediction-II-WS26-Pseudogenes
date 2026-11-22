import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MODELS_DIR = Path(__file__).resolve().parent
METRICS_DIR = MODELS_DIR / "Metrics"
GRAPHS_DIR = MODELS_DIR / "Graphs"


def main():
    GRAPHS_DIR.mkdir(exist_ok=True)
    for path in sorted(METRICS_DIR.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        epochs = run["epochs"]
        epoch_numbers = [row["epoch"] for row in epochs]
        for metric in ("loss", "accuracy"):
            fig, ax = plt.subplots()
            for prefix, label in (("train", "Training"), ("val", "Validation")):
                ax.plot(epoch_numbers, [row[f"{prefix}_{metric}"] for row in epochs], label=label)
            label = metric.capitalize()
            ax.set(title=f"{label}: {path.stem}", xlabel="Epoch", ylabel=label)
            ax.legend()
            fig.savefig(GRAPHS_DIR / f"{path.stem}-{metric}.png", bbox_inches="tight")
            plt.close(fig)

        lines = [
            fill(f"{key}: {value}", width=80, subsequent_indent="  ")
            for key, value in run["hparams"].items()
            if key not in {"dataset_path", "checkpoint_dir", "metrics_dir"}
        ]
        params = "\n".join(lines)
        width, height = 560, 44 + 14 * len(params.splitlines())
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        inset_x, inset_y = 4 / width, 4 / height
        fig.add_artist(Rectangle(
            (inset_x, inset_y), 1 - 2 * inset_x, 1 - 2 * inset_y,
            transform=fig.transFigure, fill=False, edgecolor="black",
        ))
        fig.text(0.5, 1 - 12 / height, f"Hparams: {path.stem}", ha="center", va="top")
        fig.text(12 / width, 1 - 34 / height, params, va="top", family="monospace", fontsize=8)
        fig.savefig(GRAPHS_DIR / f"{path.stem}-hparams.png", dpi=fig.dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
