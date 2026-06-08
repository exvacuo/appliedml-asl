import tensorflow as tf
from pathlib import Path
from asl_detector.constants import IMAGE_SIZE, SEED, AUGMENTED_COPIES_PER_IMAGE




# ── KNN baseline preprocessing ──────────────────────────────────────────────


def preprocess_knn(images, labels, size=64):
    """Preprocess a batch of images for the KNN baseline.

    Pipeline: grayscale → resize → normalize [0,1] → flatten.
    """
    images = tf.image.rgb_to_grayscale(images)
    images = tf.image.resize(images, (size, size))
    images = images / 255.0
    images = tf.reshape(images, (tf.shape(images)[0], size * size))
    return images, labels


# ── Extra augmentation functions ────────────────────────────────────────────

def random_gaussian_noise(images):
    """Add Gaussian noise to images while preserving the pixel value range."""
    noise = tf.random.normal(tf.shape(images), stddev=5.0, dtype=images.dtype)
    return tf.clip_by_value(images + noise, 0.0, 255.0)


# ── MobileNetV2 preprocessing ───────────────────────────────────────────────



def preprocess_mobilenet(images, labels):
    """Preprocess a batch of images for MobileNetV2.

    Resizes to 224×224 and normalises pixel values from [0, 255] to [-1, 1]
    using MobileNetV2's built-in preprocess_input.
    """
    images = tf.image.resize(images, IMAGE_SIZE)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels


# ── Data augmentation (train split only) ─────────────────────────────────────


def create_augmentation_layer():
    """Return a reproducible Sequential augmentation layer.

    Augmentations:
      - Random rotation  ±18°
      - Random translation  10% horizontal/vertical
      - Random zoom  10%
      - Random brightness  ±20%
      - Random contrast  20%
      - Gaussian noise
    """
    tf.keras.utils.set_random_seed(SEED)

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05, seed=SEED),
            tf.keras.layers.RandomTranslation(
                0.1,
                0.1,
                fill_mode="constant",
                fill_value=0.0,
                seed=SEED + 1,
            ),
            tf.keras.layers.RandomZoom(
                0.1,
                fill_mode="constant",
                fill_value=0.0,
                seed=SEED + 2,
            ),
            tf.keras.layers.RandomBrightness(0.2, seed=SEED + 3),
            tf.keras.layers.RandomContrast(0.2, seed=SEED + 4),
            tf.keras.layers.Lambda(random_gaussian_noise),
        ],
        name="augmentation",
    )

augmenter = create_augmentation_layer()


def augment_dataset(images, labels):
    augmented_batches = [
        augmenter(images, training=True)
        for _ in range(AUGMENTED_COPIES_PER_IMAGE)
    ]
    augmented_images = tf.concat(augmented_batches, axis=0)
    augmented_labels = tf.tile(labels, [AUGMENTED_COPIES_PER_IMAGE])
    return augmented_images, augmented_labels
