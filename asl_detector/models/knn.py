from asl_detector.data.dataloader import load_data, SEED
from asl_detector.data.preprocess_data import preprocess_knn
from asl_detector.features.extract_features import extract_features
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pathlib as Path
import tensorflow as tf

N_FOLDS = 3
KNN_K = 3
KNN_METRIC = "euclidean"
KNN_ALGORITHM = "ball_tree"


def run_kfold(
        X,
        y,
        n_folds: int = N_FOLDS,
        k: int = KNN_K,
        metric: str = KNN_METRIC,
        algorithm: str = KNN_ALGORITHM,
) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_accuracies, fold_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[val_idx], y[val_idx]

        knn = KNeighborsClassifier(
            n_neighbors=k, metric=metric, algorithm=algorithm, n_jobs=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        fold_accuracies.append(acc)
        fold_f1s.append(f1)

        print("accuracy: ", acc)
    return {
        "fold_accuracies": fold_accuracies,
        "fold_f1s": fold_f1s,
        "mean_accuracy": np.mean(fold_accuracies),
        "mean_f1": np.mean(fold_f1s),
    }

def train_final_model(
        X,
        y,
        k: int = KNN_K,
        metric: str = KNN_METRIC,
        algorithm: str = KNN_ALGORITHM,
) -> KNeighborsClassifier:
    """Train a single KNN on the full training set for inference/test evaluation."""
    knn = KNeighborsClassifier(
        n_neighbors=k, metric=metric, algorithm=algorithm, n_jobs=1
    )
    knn.fit(X, y)
    return knn


def evaluate_on_test(model: KNeighborsClassifier, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, average="weighted"),
    }


train, val, test = load_data(baseline=True)
X_train, y_train = extract_features(dataset=train)
K_fold = run_kfold(X_train, y_train)
final_model = train_final_model(X_train, y_train)
X_test, y_test = extract_features(dataset=test)
test_results = evaluate_on_test(final_model, X_test, y_test)
print("KNN K-fold results:", K_fold)
print("KNN test results:", test_results)
