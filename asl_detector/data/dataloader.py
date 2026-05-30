from pathlib import Path

import tensorflow as tf


DATA_DIR = Path("data/curated/train")
TEST_DATA_DIR = Path("data/curated/test")
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
            "Try downloading with uv run scripts/craft_dataset.py"
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
    test_data_dir: Path = TEST_DATA_DIR,
    baseline: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset | None, tf.data.Dataset]:
    """Return datasets from the curated train/test split."""

    if baseline:
        train_ds = create_dataset(data_dir=data_dir, batch_size=batch_size)
        test_ds = create_dataset(data_dir=test_data_dir, shuffle=False, batch_size=batch_size)

        return train_ds, None, test_ds

    train_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="training",
        batch_size=batch_size,
    )

    remaining_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="validation",
        batch_size=batch_size,
    )

    test_ds = create_dataset(data_dir=test_data_dir, shuffle=False, batch_size=batch_size)

    return train_ds, remaining_ds, test_ds
