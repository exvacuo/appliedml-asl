from asl_detector.data.dataloader import create_dataset
from asl_detector.data.preprocess_data import preprocess_knn
import numpy as np

PREPROCESS_SIZE = 32

def extract_features(size: int = PREPROCESS_SIZE) -> tuple[np.ndarray, np.ndarray]:
   
    dataset = create_dataset(shuffle=False, batch_size=256)
 
    all_X, all_y = [], []
    for batch_images, batch_labels in dataset:
        X_batch, y_batch = preprocess_knn(batch_images, batch_labels, size=size)
        all_X.append(X_batch.numpy())
        all_y.append(y_batch.numpy())
 
    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int32)

    return X, y