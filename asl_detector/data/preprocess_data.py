from asl_detector.data.dataloader import DATA_DIR
from imagededup.methods import PHash
import os
from pathlib import Path


# ── Find near duplicates ──────────────────────────────────────────────

print("hello world")
"""
phasher = PHash()
image_dir = "data/raw/test/asl_alphabet_test"
print("Encoding images...")
encodings = phasher.encode_images(image_dir)
print(f"Encoded {len(encodings)} images. Finding duplicates...")
duplicates = phasher.find_duplicates(encoding_map=encodings)


print("Near-duplicate images:", len(duplicates))
"""

# ── KNN baseline preprocessing ──────────────────────────────────────────────


def preprocess_knn(images, labels, size=64):
    """Preprocess a batch of images for the KNN baseline.

    Pipeline: crop border → grayscale → resize → normalize [0,1] → flatten.
    """
    images = crop_blue_frame(images)
    images = tf.image.rgb_to_grayscale(images)
    images = tf.image.resize(images, (size, size))
    images = images / 255.0
    images = tf.reshape(images, (tf.shape(images)[0], size * size))
    return images, labels


def crop_blue_frame(images, pixels=3):
    """Crop the border from a batch of images."""
    return images[:, pixels:-pixels, pixels:-pixels, :]




# ── MobileNetV2 preprocessing ───────────────────────────────────────────────

MOBILENET_INPUT_SIZE = (224, 224)


def preprocess_mobilenet(images, labels):
    """Preprocess a batch of images for MobileNetV2.

    Resizes to 224×224 and normalises pixel values from [0, 255] to [-1, 1]
    using MobileNetV2's built-in preprocess_input.
    """
    images = crop_blue_frame(images)
    images = tf.image.resize(images, MOBILENET_INPUT_SIZE)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels


# ── Data augmentation (train split only) ─────────────────────────────────────


def create_augmentation_layer():
    """Return a Sequential augmentation layer with mild transforms.

    Augmentations (per proposal):
      - Random rotation  ±10°
      - Random translation  10% horizontal/vertical
      - Random zoom  10%
      - Random brightness  ±10%
      - Random contrast  10%
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(10 / 360),      # ±10°
        tf.keras.layers.RandomTranslation(0.1, 0.1),   # 10% shift
        tf.keras.layers.RandomZoom(0.1),                # 10% zoom
        tf.keras.layers.RandomBrightness(0.1),          # ±10% brightness
        tf.keras.layers.RandomContrast(0.1),            # ±10% contrast
    ])