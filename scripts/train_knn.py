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

def run_baseline(
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

        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm=algorithm,n_jobs=1)
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

X,y = extract_features()
results = run_baseline(X, y)
print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
print(f"Mean F1 Score: {results['mean_f1']:.4f}")
