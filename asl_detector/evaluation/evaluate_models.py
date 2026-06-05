import json
from pathlib import Path


def _history_data(history) -> dict:
    return history.history if hasattr(history, "history") else history


def save_accuracy_history(history, output_path: Path) -> list[dict]:
    """Save training and validation accuracy per epoch as JSON."""
    data = _history_data(history)
    if "accuracy" not in data:
        raise ValueError("Training history does not contain 'accuracy'.")
    if "val_accuracy" not in data:
        raise ValueError("Training history does not contain 'val_accuracy'.")

    serializable = [
        {
            "epoch": epoch,
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
        }
        for epoch, (train_accuracy, val_accuracy) in enumerate(
            zip(data["accuracy"], data["val_accuracy"]),
            start=1,
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2))
    return serializable


def _load_accuracy_history(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation history: {path}")

    history = json.loads(path.read_text())
    if not history:
        raise ValueError(f"Evaluation history is empty: {path}")

    required = {"epoch", "train_accuracy", "val_accuracy", "train_loss", "val_loss"}
    for index, row in enumerate(history, start=1):
        missing = required.difference(row)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"{path} row {index} is missing: {missing_keys}")

    return history


def make_graphs(evaluation_dir: Path, output_dir: Path) -> list[Path]:
    """Create accuracy and loss plots from saved phase evaluation JSON files."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    for history_path in sorted(evaluation_dir.glob("*_accuracy.json")):
        phase_name = history_path.stem.replace("_accuracy", "")
        history = _load_accuracy_history(history_path)
        epochs = [row["epoch"] for row in history]

        accuracy_path = output_dir / f"{phase_name}_accuracy.png"
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
        plt.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="Validation")
        plt.title(f"{phase_name.title()} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(accuracy_path, dpi=160)
        plt.close()
        plot_paths.append(accuracy_path)

        loss_path = output_dir / f"{phase_name}_loss.png"
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
        plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation")
        plt.title(f"{phase_name.title()} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=160)
        plt.close()
        plot_paths.append(loss_path)

    if not plot_paths:
        raise FileNotFoundError(f"No *_accuracy.json files found in {evaluation_dir}")

    return plot_paths
