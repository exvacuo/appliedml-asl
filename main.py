import tensorflow as tf

from asl_detector.data.dataloader import load_data
from asl_detector.models.knn import run_kfold, train_final_model, evaluate_on_test
from asl_detector.features import extract_features
from asl_detector.data.preprocess_data import preprocess_knn


class Pipeline:
    """Class that encapsulates the pipeline for the ASL classification task."""

    def __init__(self) -> None:
        self.train = None
        self.val = None
        self.test = None

    def run_baseline(self, extract_features):
        """Load data, preprocess for KNN, and trains with Kfold cross validation"""
        train, val, test = load_data(baseline=True)
        X_train, y_train = extract_features(dataset=train)
        K_fold = run_kfold(X_train, y_train)
        
        return train_results

    


if __name__ == "__main__":
    pipeline = Pipeline()
    results = pipeline.train_baseline(extract_features=extract_features)
