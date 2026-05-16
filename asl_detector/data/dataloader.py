from pathlib import Path

import tensorflow as tf


DATA_DIR = Path("data/raw/train/asl_alphabet_train")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# TODO: check if datasplit works properly
# TODO: check and alter main further to fit the new load_data


def create_dataset(
    data_dir: Path = DATA_DIR,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float | None = None,  # use this for dataset split
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


def split_dataset(data_dir: Path = DATA_DIR) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """ Return the train, validation and test TensorFlow datasets using create_dataset. Split of 80-10-10 respectively.
    """
    train_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
    )

    temp_ds = create_dataset(
        data_dir=data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        shuffle=True,
    )

    # split temporary dataset into validation and test set
    val_ds = temp_ds.take(int(len(temp_ds) * 0.5))
    test_ds = temp_ds.skip(int(len(temp_ds) * 0.5))

    # batch validation and test set
    val_ds = val_ds.batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds


def load_data(data_dir: Path = DATA_DIR) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return the split training, validation and test sets."""
    return split_dataset(data_dir=data_dir)

    # Or if we only want to return the training and validation set:
    # train_ds = create_dataset(
    #     data_dir=data_dir,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=SEED,
    # )

    # val_ds = create_dataset(
    #     data_dir=data_dir,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=SEED,
    #     shuffle=True,
    # )

    # return train_ds, val_ds
