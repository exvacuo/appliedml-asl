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
