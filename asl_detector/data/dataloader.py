from pathlib import Path

import tensorflow as tf


DATA_DIR = Path("data/raw/train/asl_alphabet_train")
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
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return train (80%), val (10%), test (10%) datasets.

    Uses create_dataset's validation_split to carve out 20%, then splits
    that remainder in half for val and test.  The remaining 20% is loaded
    unbatched so we can split by sample count, then re-batched.
    """
    train_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="training",
    )

    # Load remaining 20% unbatched so we can split cleanly
    remaining_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="validation",
        batch_size=None,
        shuffle=False,
    )

    # Split into equal halves → 10% val, 10% test
    n_remaining = remaining_ds.cardinality()
    if n_remaining == tf.data.UNKNOWN_CARDINALITY:
        n_remaining = sum(1 for _ in remaining_ds)
    half = int(n_remaining) // 2

    val_ds = remaining_ds.take(half).batch(BATCH_SIZE)
    test_ds = remaining_ds.skip(half).batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds
