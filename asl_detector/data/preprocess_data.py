import tensorflow as tf


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


# ── Extra augmentation functions ────────────────────────────────────────────

def random_hue(images):
    """Randomly shift image hue."""
    return tf.image.random_hue(images, max_delta=0.25)


def random_invert(images):
    """Randomly invert about half of the images in a batch."""
    random_values = tf.random.uniform((tf.shape(images)[0], 1, 1, 1))
    inverted_images = 255.0 - images
    return tf.where(random_values < 0.5, inverted_images, images)


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
    """Return a Sequential augmentation layer with stronger transforms.

    Augmentations:
      - Random rotation  ±25°
      - Random translation  20% horizontal/vertical
      - Random zoom  25%
      - Random brightness  ±30%
      - Random contrast  40%
      - Random hue shift
      - Random color inversion
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(25 / 360),      # ±25°
        tf.keras.layers.RandomTranslation(0.2, 0.2),   # 20% shift
        tf.keras.layers.RandomZoom(0.25),              # 25% zoom
        tf.keras.layers.RandomBrightness(0.3),         # ±30% brightness
        tf.keras.layers.RandomContrast(0.4),           # stronger contrast
        tf.keras.layers.Lambda(random_hue),            # stronger color shift
        tf.keras.layers.Lambda(random_invert),         # invert about half the images
    ])