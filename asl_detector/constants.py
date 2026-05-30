from pathlib import Path

# Paths
DATA_RAW_BASE_DIR = Path("data/raw")
DATA_RAW_TRAIN_DIR = DATA_RAW_BASE_DIR / "train" / "asl_alphabet_train"
ZIP_PATH = DATA_RAW_BASE_DIR / "asl-alphabet.zip"
KAGGLE_DATASET = "grassknoted/asl-alphabet"

DATA_DEDUPLICATED_DIR = Path("data/deduplicated/train")
DATA_CROPPED_DIR = Path("data/cropped")
DATA_CURATED_TRAIN_DIR = Path("data/curated/train")
DATA_CURATED_TEST_DIR = Path("data/curated/test")

# Dataset Information
CLASSES = [
    "A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"
]

# Data Processing Params
SEED = 42
TEST_RATIO = 0.2
BALANCE_THRESHOLD_EXCLUDED_CLASSES = {"nothing"}
OVERSAMPLE_UNDER_TARGET_CLASSES = {"nothing"}

# Image Processing Params
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
PIXELS_TO_CROP = 3
