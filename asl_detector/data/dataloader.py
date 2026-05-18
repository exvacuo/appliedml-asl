from pathlib import Path
from typing import Callable
import numpy as np

import tensorflow as tf


DATA_DIR = Path("data/raw/train/asl_alphabet_train")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# TODO: data splitting rather than making a single dataset

def create_dataset(
    data_dir: Path = DATA_DIR,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float | None = None, # use this for dataset split
    subset: str | None = None,
    seed: int = SEED,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from class folders."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Could not find the data at {data_dir}. "
            "Try downloading with python scripts/download_data.py"
        )

    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
    )



def load_data(data_dir: Path = DATA_DIR, validation_split: float | None = None, subset: str | None = None, shuffle: bool = True) -> tf.data.Dataset:
    """Return full dataset"""
    dataset = create_dataset(data_dir=data_dir, validation_split=validation_split, subset=subset, shuffle=shuffle)
    return dataset
