import tensorflow as tf

def preprocess_knn(images, labels, size=64):
    """Preprocess a batch of images for the KNN baseline."""
    images = crop_blue_frame(images)
    images = tf.image.rgb_to_grayscale(images)
    images = tf.image.resize(images, (size, size))
    images = images / 255.0
    images = tf.reshape(images, (tf.shape(images)[0], size * size))
    return images, labels


def crop_blue_frame(images, pixels=3):
    """Crop the border from a batch of images."""
    return images[:, pixels:-pixels, pixels:-pixels, :]

# TODO: add other preprocessing functions here

## Data Augmentation




