import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from asl_detector.constants import (
    BATCH_SIZE,
    CLASSES,
    DATA_CURATED_TEST_DIR,
    HF_MODEL_FILENAME,
    HF_MODEL_REPO_ID,
    IMAGE_SIZE,
)
from asl_detector.data.dataloader import create_dataset
from asl_detector.data.preprocess_data import preprocess_mobilenet
from asl_detector.models.mobilenetv2 import ASLMobilenetv2


def load_final_model():
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
    model = ASLMobilenetv2(num_classes=len(CLASSES), dropout_rate=0.2).model
    model.load_weights(weights_path)
    return model, weights_path


def load_test_dataset(data_dir: Path = DATA_CURATED_TEST_DIR):
    ds = create_dataset(data_dir=data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    return ds.map(preprocess_mobilenet), list(ds.file_paths)


def predict_dataset(model, dataset):
    probabilities = model.predict(dataset)
    y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
    y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0).astype(np.int32)
    return y_true, y_pred


def make_confusion_matrix(y_true, y_pred, labels=CLASSES):
    indices = [CLASSES.index(label) for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    return {"labels": labels, "matrix": matrix.astype(int).tolist()}


def plot_confusion_matrix(confusion: dict, output_path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    labels = confusion["labels"]
    matrix = np.array(confusion["matrix"])
    size = max(4, min(14, len(labels) * 0.45))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def measure_inference_time_knn(model, X_test, repeats: int = 3) -> float:
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X_test)
        seconds.append(time.perf_counter() - start)
    return float(np.mean(seconds))

def measure_inference_time(model, dataset, repeats: int = 3) -> dict:
    seconds = []
    n_images = 0
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(dataset, verbose=0)
        seconds.append(time.perf_counter() - start)
        if n_images == 0:
            n_images = sum(int(labels.shape[0]) for _, labels in dataset)
    mean_seconds = float(np.mean(seconds))
    return {
        "repeats": repeats,
        "images_per_run": n_images,
        "mean_seconds_per_run": mean_seconds,
        "mean_ms_per_image": mean_seconds / n_images * 1000,
    }


def evaluate_test_data(
    data_dir: Path = DATA_CURATED_TEST_DIR,
    confusion_labels: list[str] = CLASSES,
    inference_repeats: int = 3,
) -> dict:
    model, weights_path = load_final_model()
    dataset, file_paths = load_test_dataset(data_dir)
    y_true, y_pred = predict_dataset(model, dataset)
    return {
        "model": "mobilenetv2",
        "weights_path": str(weights_path),
        "test_data_dir": str(data_dir),
        "num_test_images": len(file_paths),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": make_confusion_matrix(y_true, y_pred, confusion_labels),
        "inference_time": measure_inference_time(model, dataset, inference_repeats),
    }


def save_test_evaluation(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{results['model']}_test_evaluation.json"
    output_path.write_text(json.dumps(results, indent=2))
    return output_path


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


if __name__ == "__main__":
    results = evaluate_test_data()
    path = save_test_evaluation(results, Path("models/evaluation"))
    print(f"Wrote {path}")
