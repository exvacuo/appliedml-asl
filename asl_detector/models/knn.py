from asl_detector.data.dataloader import load_data, SEED, BATCH_SIZE
from asl_detector.data.preprocess_data import create_augmentation_layer, crop_blue_frame
from asl_detector.features.extract_features import extract_features
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import cv2
import numpy as np
import tensorflow as tf

N_FOLDS = 3
KNN_K = 3
KNN_METRIC = "euclidean"
KNN_ALGORITHM = "brute"
PCA_COMPONENTS = 50



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

        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

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
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    X_pca = pca.fit_transform(X)

    knn = KNeighborsClassifier(
        n_neighbors=k, metric=metric, algorithm=algorithm, n_jobs=1
    )
    knn.fit(X_pca, y)
    return knn, pca


def evaluate_on_test(model: KNeighborsClassifier, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, average="weighted"),
    }

## Load splits
train, val, test = load_data(baseline=True)
test = test.batch(BATCH_SIZE)


print("Augmenting training data...")
## Augmentation
augmenter = create_augmentation_layer()

# Preview one unmodified image, the blue-frame-cropped image, and several augmented versions with OpenCV.
for images, labels in train.take(1):
    unmodified_image = images[0]
    cropped_images = crop_blue_frame(images)
    image = cropped_images[0]
    label = labels[0]

    unmodified_display = unmodified_image.numpy().astype("uint8")
    unmodified_display = cv2.cvtColor(unmodified_display, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"Unmodified - Label: {label.numpy()}", unmodified_display)

    cropped_display = image.numpy().astype("uint8")
    cropped_display = cv2.cvtColor(cropped_display, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"Blue Frame Removed - Label: {label.numpy()}", cropped_display)

    for i in range(4):
        augmented_image = augmenter(tf.expand_dims(image, axis=0), training=True)[0]
        augmented_image = np.clip(augmented_image.numpy(), 0, 255).astype("uint8")
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Blue Frame Removed + Augmented {i + 1}", augmented_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break

def augment_dataset(images, labels):
    images = crop_blue_frame(images)
    return augmenter(images), labels

train_combined = train.concatenate(train.map(augment_dataset))


## Extract features
print("Extracting features for KNN...")
X_train, y_train = extract_features(dataset=train_combined)
X_test, y_test = extract_features(dataset=test)

## Run Kfold
print("Running KNN with K-fold cross validation...")
K_fold = run_kfold(X_train, y_train)
print("KNN K-fold results:", K_fold)

## Train final model
print("Training final KNN model on full training set...")
final_model, pca = train_final_model(X_train, y_train)
X_test = pca.transform(X_test)
test_results = evaluate_on_test(final_model, X_test, y_test)
print("KNN test results:", test_results)
