import tensorflow as tf
from asl_detector.constants import CHECKPOINT_DIR, MODEL_DIR, MODEL_WEIGHTS_PATH,SEED
from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_mobilenet, augment_dataset
from asl_detector.models.mobilenetv2 import ASLMobilenetv2
from asl_detector.models.train.tune import find_hyperparameters


def setup_gpu():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("WARNING: No GPU detected. Training will run on CPU (very slow).")
        return

    print(f"GPU detected: {gpus}")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU already initialized; skipping memory_growth: {e}")


def training_callbacks(phase_name):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(CHECKPOINT_DIR / f"backup_{phase_name}"),
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / f"{phase_name}_best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]


def checkpoint_path(phase_name):
    return CHECKPOINT_DIR / f"{phase_name}_best.weights.h5"


def done_path(phase_name):
    return CHECKPOINT_DIR / f"{phase_name}.done"


def train_model():
    setup_gpu()
    hyperparams_phase_1, hyperparams_phase_2 = find_hyperparameters()

    train, val, test = load_data(batch_size=hyperparams_phase_1["batch_size"], baseline=False)
    train_combined = train.concatenate(train.map(augment_dataset))
    train_combined = (
         train_combined
         .unbatch()
         .shuffle(buffer_size=1000, seed=SEED)
         .batch(hyperparams_phase_1["batch_size"])
         ) 

    train_combined = train_combined.map(preprocess_mobilenet)

    val_preprocessed = val.map(preprocess_mobilenet)
    test_preprocessed = test.map(preprocess_mobilenet)


    ## Phase 1
    model = ASLMobilenetv2(num_classes=29, dropout_rate=hyperparams_phase_1["dropout_rate"])
    if done_path("phase1").exists() and checkpoint_path("phase1").exists():
        model.model.load_weights(checkpoint_path("phase1"))
        print(f"Loaded Phase 1 checkpoint from {checkpoint_path('phase1')}")
    else:
        model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams_phase_1["learning_rate"]),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        model.model.fit(train_combined, validation_data=val_preprocessed, epochs=30,
                callbacks=training_callbacks("phase1"))
        done_path("phase1").touch()
    
    ## Phase 2
    if done_path("phase2").exists() and checkpoint_path("phase2").exists():
        model.unfreeze_top_n_layers(hyperparams_phase_2["n_layers"])
        model.model.load_weights(checkpoint_path("phase2"))
        print(f"Loaded Phase 2 checkpoint from {checkpoint_path('phase2')}")
    else:
        model.model.load_weights(checkpoint_path("phase1"))
        print(f"Loaded Phase 1 checkpoint from {checkpoint_path('phase1')}")
        model.unfreeze_top_n_layers(hyperparams_phase_2["n_layers"])
        model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams_phase_2["learning_rate"]),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        model.model.fit(train_combined, validation_data=val_preprocessed, epochs=20,
                callbacks=training_callbacks("phase2"))
        done_path("phase2").touch()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.model.save_weights(MODEL_WEIGHTS_PATH)
    print(f"Saved model weights to {MODEL_WEIGHTS_PATH}")
    test_loss, test_acc = model.model.evaluate(test_preprocessed)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    train_model()
