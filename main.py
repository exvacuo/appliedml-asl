import tensorflow as tf

from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_knn


class Pipeline:
    """Class that encapsulates the pipeline for the ASL classification task."""

    def __init__(self) -> None:
        self.train = None
        self.val = None
        self.test = None

    def run_baseline(self):
        """Load data, preprocess for KNN, and return train/val/test."""
        self.train, self.val, self.test = load_data()
        self.train = self.train.map(
            lambda images, labels: preprocess_knn(images, labels, size=64)
        )
        return self.train, self.val, self.test

if __name__ == "__main__":
    pipeline = Pipeline()
    train, val, test = pipeline.run_baseline()

    train_n = train.cardinality().numpy() * 32
    val_n   = val.cardinality().numpy()   * 32
    test_n  = test.cardinality().numpy()  * 32
    total   = train_n + val_n + test_n

    print(f"Train samples: {train_n} ({train_n/total:.1%})")
    print(f"Val samples:   {val_n} ({val_n/total:.1%})")
    print(f"Test samples:  {test_n} ({test_n/total:.1%})") 
    
