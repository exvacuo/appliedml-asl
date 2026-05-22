from pathlib import Path
from typing import Callable
import numpy as np

import tensorflow as tf


DATA_DIR = Path("data/deduplicated/train")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def create_dataset(
    data_dir: Path = DATA_DIR,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float | None = None,
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


def load_data(
    data_dir: Path = DATA_DIR,
    baseline: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset | None, tf.data.Dataset]:
    """Return train (80%), val (10%), test (10%) datasets or 
    80/20 train/test split for KNN baseline.
    """

    train_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="training",
    )

    remaining_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="validation",
        batch_size=None,
        shuffle=False,
    )

    if baseline == False:
        n_remaining = remaining_ds.cardinality()
        if n_remaining == tf.data.UNKNOWN_CARDINALITY:
            n_remaining = sum(1 for _ in remaining_ds)
        half = int(n_remaining) // 2

        val_ds = remaining_ds.take(half).batch(BATCH_SIZE)
        test_ds = remaining_ds.skip(half).batch(BATCH_SIZE)

        return train_ds, val_ds, test_ds

    test_ds = remaining_ds

    return train_ds, None, test_ds
