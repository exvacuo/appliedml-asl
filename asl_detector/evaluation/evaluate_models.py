import json
from pathlib import Path


def _history_data(history) -> dict:
    return history.history if hasattr(history, "history") else history


def save_accuracy_history(history, output_path: Path) -> list[dict]:
    """Save training and validation accuracy and loss per epoch as JSON."""
    data = _history_data(history)
    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        if key not in data:
            raise ValueError(f"Training history does not contain '{key}'.")

    serializable = [
        {
            "epoch": epoch,
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        for epoch, (train_acc, val_acc, train_loss, val_loss) in enumerate(
            zip(data["accuracy"], data["val_accuracy"], data["loss"], data["val_loss"]),
            start=1,
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2))
    return serializable
