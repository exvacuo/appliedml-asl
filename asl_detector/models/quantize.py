import tensorflow as tf
from huggingface_hub import hf_hub_download

from asl_detector.constants import (
    HF_MODEL_FILENAME,
    HF_MODEL_REPO_ID,
    MODEL_DIR,
    MODEL_WEIGHTS_PATH,
    QUANTIZED_WEIGHTS_PATH,
)
from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_mobilenet
from asl_detector.models.mobilenetv2 import ASLMobilenetv2


def download_weights():
    if not MODEL_WEIGHTS_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading weights from Hugging Face ")
        hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=HF_MODEL_FILENAME,
            local_dir=str(MODEL_DIR),
        )
        
def calibration_data(num_samples: int = 200):
    _, val_ds, _ = load_data(baseline=False)
    calibration_ds = (
        val_ds.unbatch()
        .shuffle(buffer_size=1000)
        .take(num_samples)
        .map(lambda image, label: preprocess_mobilenet(image, label)[0])        
        .batch(1)
    )
    def callibrate():
        for image_batch in calibration_ds:
            yield [image_batch]

    return callibrate



def quantize_model():
    download_weights()
    model = ASLMobilenetv2()
    model.model.load_weights(MODEL_WEIGHTS_PATH)

    saved_model_path = MODEL_DIR / "saved_model"
    model.model.export(saved_model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_data()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    quantized_model = converter.convert()

    QUANTIZED_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUANTIZED_WEIGHTS_PATH.write_bytes(quantized_model)
    print(f"Quantized model saved to {QUANTIZED_WEIGHTS_PATH}")




if __name__ == "__main__":
    quantize_model()
