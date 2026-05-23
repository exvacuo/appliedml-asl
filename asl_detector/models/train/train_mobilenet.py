from pathlib import Path

import tensorflow as tf
from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_mobilenet, create_augmentation_layer
from asl_detector.models.mobilenetv2 import ASLMobilenetv2

# branch created
AUGMENTED_COPIES_PER_IMAGE = 3
MODEL_DIR = Path("models")
WEIGHTS_PATH = MODEL_DIR / "mobilenetv2.weights.h5"

train, val, test = load_data(baseline=False)

augmenter = create_augmentation_layer()


def augment_dataset(images, labels):
    augmented_batches = [
        augmenter(images, training=True)
        for _ in range(AUGMENTED_COPIES_PER_IMAGE)
    ]
    augmented_images = tf.concat(augmented_batches, axis=0)
    augmented_labels = tf.repeat(labels, repeats=AUGMENTED_COPIES_PER_IMAGE, axis=0)
    print(f"Augmented batch shape: {augmented_images.shape}, Augmented labels shape: {augmented_labels.shape}")
    return augmented_images, augmented_labels

def main():
    
    train_combined = train.concatenate(train.map(augment_dataset))
    train_combined = train_combined.map(preprocess_mobilenet)
    val_preprocessed = val.map(preprocess_mobilenet)
    test_preprocessed = test.map(preprocess_mobilenet)
    model = ASLMobilenetv2(num_classes=29, dropout_rate=0.2)
    model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.model.fit(train_combined, validation_data=val_preprocessed, epochs=10)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.model.save_weights(WEIGHTS_PATH)
    print(f"Saved model weights to {WEIGHTS_PATH}")
    test_loss, test_acc = model.model.evaluate(test_preprocessed)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
