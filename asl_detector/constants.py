from pathlib import Path
import os

# Paths
DATA_RAW_BASE_DIR = Path("data/raw")
DATA_RAW_TRAIN_DIR = DATA_RAW_BASE_DIR / "train" / "asl_alphabet_train"
ZIP_PATH = DATA_RAW_BASE_DIR / "asl-alphabet.zip"
KAGGLE_DATASET = "grassknoted/asl-alphabet"

AUGMENTED_COPIES_PER_IMAGE = 3
MODEL_DIR = Path("models")
WEIGHTS_PATH = MODEL_DIR / "mobilenetv2.weights.h5"


DATA_DEDUPLICATED_DIR = Path("data/deduplicated/train")
DATA_CROPPED_DIR = Path("data/cropped")
DATA_CURATED_TRAIN_DIR = Path("data/curated/train")
DATA_CURATED_TEST_DIR = Path("data/curated/test")

# Model Paths
MODEL_DIR = Path("models")
MODEL_WEIGHTS_PATH = MODEL_DIR / "mobilenetv2.weights.h5"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "xvacuo/asl-detector")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "mobilenetv2.weights.h5")

# Dataset Information
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]

# Data Processing Params
SEED = 42
TEST_RATIO = 0.1
BALANCE_THRESHOLD_EXCLUDED_CLASSES = {"nothing"}
OVERSAMPLE_UNDER_TARGET_CLASSES = {"nothing"}

# Image Processing Params
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
PIXELS_TO_CROP = 3

# Hyperparameter Tuning Params

PHASE1_SEARCH_SPACE = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "dropout_rate": [0.2, 0.3, 0.5],
    "batch_size": [16, 32, 64],
}

PHASE2_SEARCH_SPACE = {
    "learning_rate": [0.001, 0.0001, 0.00001],
    "n_layers": [ 25, 50, 75],
}
