import tensorflow as tf

from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_knn


class Pipeline:
    """Class that encapsulates the pipeline for the ASL classification task."""

    def __init__(self) -> None:
        self.data = None

    def run_baseline(self):
        self.data = load_data()
        self.data = self.data.map(
            lambda images, labels: preprocess_knn(images, labels, size=64)
        )
        return self.data


if __name__ == "__main__":
    pipeline = Pipeline()
    dataset = pipeline.run_baseline()

    images, labels = next(iter(dataset))

    print(images.shape)

    # remove this, it's for a reference of the preprocessing
    reference_image = tf.reshape(images[0], (64, 64, 1))
    reference_image = tf.cast(reference_image * 255.0, tf.uint8)
    encoded_image = tf.io.encode_png(reference_image)
    tf.io.write_file("reference_image.png", encoded_image) # for reference