import tensorflow as tf

from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_knn

# TODO: check run_baseline and main to fit with dataloader


class Pipeline:
    """Class that encapsulates the pipeline for the ASL classification task."""

    def __init__(self) -> None:
        self.data = None

    def run_baseline(self):
        # self.data = load_data()
        # self.data = self.data.map(
        #     lambda images, labels: preprocess_knn(images, labels, size=64)
        # )

        # this would fit with the new implementation of data_load
        self.train, self.val, self.test = load_data()
        self.train = self.train.map(
            lambda images, labels: preprocess_knn(images, labels, size=64))

        # return train and val
        return self.train, self.val


if __name__ == "__main__":
    pipeline = Pipeline()
    # dataset = pipeline.run_baseline()
    # images, labels = next(iter(dataset))

    train, val = pipeline.run_baseline()
    images, labels = next(iter(train))
    images, labels = next(iter(val))

    print(images.shape)

    # remove this, it's for a reference of the preprocessing
    reference_image = tf.reshape(images[0], (64, 64, 1))
    reference_image = tf.cast(reference_image * 255.0, tf.uint8)
    encoded_image = tf.io.encode_png(reference_image)
    tf.io.write_file("reference_image.png", encoded_image)  # for reference
