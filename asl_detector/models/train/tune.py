import optuna
from asl_detector.constants import CHECKPOINT_DIR, PHASE1_SEARCH_SPACE, PHASE2_SEARCH_SPACE, SEED

import tensorflow as tf
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState


from asl_detector.models.mobilenetv2 import ASLMobilenetv2
from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_mobilenet, augment_dataset


best_weights = None
best_val_accuracy = 0.0
PHASE1_TRIALS = 27
PHASE2_TRIALS = 9


def _checkpoint_name(phase_name, **params):
    suffix = "_".join(f"{key}_{value}" for key, value in sorted(params.items()))
    return f"tune_{phase_name}_{suffix}".replace(".", "p")


def _trial_callbacks(checkpoint_name):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(CHECKPOINT_DIR / f"backup_{checkpoint_name}"),
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / f"{checkpoint_name}.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    return callbacks


def _remaining_trials(study, total_trials):
    completed_trials = [
        trial for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]
    return max(0, total_trials - len(completed_trials))


def _load_best_phase1_weights(study, dropout_rate):
    global best_weights
    if best_weights is not None:
        return

    checkpoint_path = CHECKPOINT_DIR / (
        _checkpoint_name("phase1", **study.best_trial.params) + ".weights.h5"
    )
    if not checkpoint_path.exists():
        return

    model = ASLMobilenetv2(dropout_rate=dropout_rate)
    model.model.load_weights(checkpoint_path)
    best_weights = model.model.get_weights()

### Phase 1: Tune the learning rate + Dropout rate for the classifier head, with the backbone frozen.
def objective(trial):
    global best_weights, best_val_accuracy
    tf.keras.backend.clear_session()

    learning_rate = trial.suggest_categorical("learning_rate", PHASE1_SEARCH_SPACE["learning_rate"])
    dropout_rate = trial.suggest_categorical("dropout_rate", PHASE1_SEARCH_SPACE["dropout_rate"])
    batch_size = trial.suggest_categorical("batch_size", PHASE1_SEARCH_SPACE["batch_size"])
    checkpoint_name = _checkpoint_name(
        "phase1",
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
    )

    train_ds, val_ds, _ = load_data(batch_size=batch_size, baseline=False)

    train_combined = train_ds.concatenate(train_ds.map(augment_dataset))
    train_combined = train_combined.map(preprocess_mobilenet)
    val_preprocessed = val_ds.map(preprocess_mobilenet)

    model = ASLMobilenetv2(dropout_rate=dropout_rate)

    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
        run_eagerly=True,
    )

    model.model.fit(
        train_combined,
        validation_data=val_preprocessed,
        epochs=20,
        callbacks=_trial_callbacks(checkpoint_name),
    )              
    _, val_accuracy = model.model.evaluate(val_preprocessed)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_weights = model.model.get_weights()
    return val_accuracy   



### Phase 2: Unfreeze the last N layers of the backbone and tune N, learning rate, and dropout rate together.
def objective_phase2(trial, dropout_rate, batch_size, weights):
    tf.keras.backend.clear_session() 

    learning_rate = trial.suggest_categorical("learning_rate", PHASE2_SEARCH_SPACE["learning_rate"])
    n_layers = trial.suggest_categorical("n_layers", PHASE2_SEARCH_SPACE["n_layers"])
    checkpoint_name = _checkpoint_name(
        "phase2",
        learning_rate=learning_rate,
        n_layers=n_layers,
    )

    train_ds, val_ds, _ = load_data(batch_size=batch_size, baseline=False)
    train_combined = train_ds.concatenate(train_ds.map(augment_dataset))
    train_combined = train_combined.map(preprocess_mobilenet)
    val_preprocessed = val_ds.map(preprocess_mobilenet)



    model = ASLMobilenetv2(dropout_rate=dropout_rate, unfreeze_top_n_layers=n_layers)
    if weights is not None:
        model.model.set_weights(weights)
    

    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.model.fit(
        train_combined,
        validation_data=val_preprocessed,
        epochs=20,
        callbacks=_trial_callbacks(checkpoint_name),
    )
              
    _, val_accuracy = model.model.evaluate(val_preprocessed)
    return val_accuracy

def find_hyperparameters():
    "" "Run Phase 1 hyperparameter tuning."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{CHECKPOINT_DIR / 'optuna.db'}",
        heartbeat_interval=60,
        grace_period=300,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
    )

    study_phase_1 = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GridSampler(PHASE1_SEARCH_SPACE, seed=SEED),
                                    pruner=optuna.pruners.MedianPruner(),
                                    storage=storage,
                                    study_name="mobilenetv2_phase1",
                                    load_if_exists=True,
                                    )
    
    study_phase_1.optimize(objective, n_trials=_remaining_trials(study_phase_1, PHASE1_TRIALS))
    hyperparams_phase_1 = study_phase_1.best_params
    print("Best hyperparameters from Phase 1:", hyperparams_phase_1)

    "" "Run Phase 2 hyperparameter tuning, using the best dropout rate and batch size from Phase 1."""
    best_dropout_rate = hyperparams_phase_1["dropout_rate"]
    best_batch_size = hyperparams_phase_1["batch_size"]
    _load_best_phase1_weights(study_phase_1, best_dropout_rate)

    study_phase_2 = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=SEED),
                                        pruner=optuna.pruners.MedianPruner(),
                                        storage=storage,
                                        study_name="mobilenetv2_phase2",
                                        load_if_exists=True,
                                        )
    study_phase_2.optimize(
        lambda trial: objective_phase2(trial, best_dropout_rate, best_batch_size, best_weights),
        n_trials=_remaining_trials(study_phase_2, PHASE2_TRIALS),
    )
    hyperparams_phase_2 = study_phase_2.best_params
    print("Best hyperparameters from Phase 2:", hyperparams_phase_2)

    return hyperparams_phase_1, hyperparams_phase_2

if __name__ == "__main__":
    hyperparams_phase_1, hyperparams_phase_2 = find_hyperparameters()
