from pathlib import Path

import numpy as np

from asl_detector.constants import CLASSES, DATA_CURATED_TEST_DIR
from asl_detector.evaluation.evaluate_models import (
    load_final_model,
    load_test_dataset,
    make_confusion_matrix,
    make_graphs,
    measure_inference_time,
    plot_confusion_matrix,
    predict_dataset,
    save_test_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "models/evaluation"
PLOTS = OUT / "plots"


def select_confusion_labels(confusion: dict, labels: list[str]) -> dict:
    matrix = np.array(confusion["matrix"])
    indices = [CLASSES.index(label) for label in labels]
    return {"labels": labels, "matrix": matrix[np.ix_(indices, indices)].tolist()}


def make_evaluation_plots() -> list[Path]:
    model, weights_path = load_final_model()
    dataset, files = load_test_dataset(PROJECT_ROOT / DATA_CURATED_TEST_DIR)
    y_true, y_pred = predict_dataset(model, dataset)
    confusion = make_confusion_matrix(y_true, y_pred)

    results = {
        "model": "mobilenetv2",
        "weights_path": str(weights_path),
        "num_test_images": len(files),
        "test_accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": confusion,
        "inference_time": measure_inference_time(model, dataset),
    }

    written_paths = [save_test_evaluation(results, OUT)]
    written_paths.append(
        plot_confusion_matrix(confusion, PLOTS / "confusion_all_classes.png", "All Classes")
    )
    written_paths.append(
        plot_confusion_matrix(
            select_confusion_labels(confusion, ["I", "J"]),
            PLOTS / "confusion_i_j.png",
            "I/J",
        )
    )
    written_paths.append(
        plot_confusion_matrix(
            select_confusion_labels(confusion, ["M", "N"]),
            PLOTS / "confusion_m_n.png",
            "M/N",
        )
    )
    written_paths.extend(make_graphs(OUT, PLOTS))
    return written_paths


def main():
    for path in make_evaluation_plots():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
