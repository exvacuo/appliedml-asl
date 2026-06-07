import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from asl_detector.constants import CLASSES, DATA_CURATED_TEST_DIR, MODEL_WEIGHTS_PATH
from asl_detector.evaluation.evaluate_models import (
    load_test_dataset,
    measure_inference_time,
    predict_dataset,
)
from asl_detector.models.mobilenetv2 import ASLMobilenetv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("models/evaluation/test_metrics.json")


def evaluate_metrics(output_path: Path = OUTPUT_PATH) -> dict:
    model = ASLMobilenetv2(num_classes=len(CLASSES)).model
    model.load_weights(PROJECT_ROOT / MODEL_WEIGHTS_PATH)

    dataset, file_paths = load_test_dataset(PROJECT_ROOT / DATA_CURATED_TEST_DIR)
    y_true, y_pred = predict_dataset(model, dataset)

    metrics = {
        "model": "mobilenetv2",
        "weights_path": str(MODEL_WEIGHTS_PATH),
        "test_data_dir": str(DATA_CURATED_TEST_DIR),
        "num_test_images": len(file_paths),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "inference_time": measure_inference_time(model, dataset, repeats=1),
    }

    output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    evaluate_metrics()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
