import optuna
from asl_detector.constants import PHASE1_SEARCH_SPACE, PHASE2_SEARCH_SPACE, SEED

import tensorflow as tf


from asl_detector.models.mobilenetv2 import ASLMobilenetv2
from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess_mobilenet, augment_dataset


### Phase 1: Tune the learning rate + Dropout rate for the classifier head, with the backbone frozen.
def objective(trial):
    tf.keras.backend.clear_session() 

    learning_rate = trial.suggest_categorical("learning_rate", PHASE1_SEARCH_SPACE["learning_rate"])
    dropout_rate = trial.suggest_categorical("dropout_rate", PHASE1_SEARCH_SPACE["dropout_rate"])
    batch_size = trial.suggest_categorical("batch_size", PHASE1_SEARCH_SPACE["batch_size"])

    train_ds, val_ds, _ = load_data(batch_size=batch_size, baseline=False)

    train_combined = train_ds.concatenate(train_ds.map(augment_dataset))
    train_combined = train_combined.map(preprocess_mobilenet)
    val_preprocessed = val_ds.map(preprocess_mobilenet)

    model = ASLMobilenetv2(dropout_rate=dropout_rate)

    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.model.fit(train_combined, validation_data=val_preprocessed, epochs=20,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                           patience=3,
                                                           restore_best_weights=True)])
              
    _, val_accuracy = model.model.evaluate(val_preprocessed)
    return val_accuracy   



### Phase 2: Unfreeze the last N layers of the backbone and tune N, learning rate, and dropout rate together.
def objective_phase2(trial, dropout_rate, batch_size):
    tf.keras.backend.clear_session() 

    learning_rate = trial.suggest_categorical("learning_rate", PHASE2_SEARCH_SPACE["learning_rate"])
    n_layers = trial.suggest_categorical("n_layers", PHASE2_SEARCH_SPACE["n_layers"])

    train_ds, val_ds, _ = load_data(batch_size=batch_size, baseline=False)
    train_combined = train_ds.concatenate(train_ds.map(augment_dataset))
    train_combined = train_combined.map(preprocess_mobilenet)
    val_preprocessed = val_ds.map(preprocess_mobilenet)



    model = ASLMobilenetv2(dropout_rate=dropout_rate, unfreeze_top_n_layers=n_layers)
    

    model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.model.fit(train_combined, validation_data=val_preprocessed, epochs=20,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                           patience=3,
                                                           restore_best_weights=True)])
              
    _, val_accuracy = model.model.evaluate(val_preprocessed)
    return val_accuracy

def find_hyperparameters():
    "" "Run Phase 1 hyperparameter tuning."""
    study_phase_1 = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GridSampler(PHASE1_SEARCH_SPACE, seed=SEED),
                                    pruner=optuna.pruners.MedianPruner()
                                    )
    
    study_phase_1.optimize(objective, n_trials=27) 
    hyperparams_phase_1 = study_phase_1.best_params
    print("Best hyperparameters from Phase 1:", hyperparams_phase_1)

    "" "Run Phase 2 hyperparameter tuning, using the best dropout rate and batch size from Phase 1."""
    best_dropout_rate = hyperparams_phase_1["dropout_rate"]
    best_batch_size = hyperparams_phase_1["batch_size"]

    study_phase_2 = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=SEED),
                                        pruner=optuna.pruners.MedianPruner()
                                        )
    study_phase_2.optimize(lambda trial: objective_phase2(trial, best_dropout_rate, best_batch_size), n_trials=9)
    hyperparams_phase_2 = study_phase_2.best_params
    print("Best hyperparameters from Phase 2:", hyperparams_phase_2)

    return hyperparams_phase_1, hyperparams_phase_2

if __name__ == "__main__":
    hyperparams_phase_1, hyperparams_phase_2 = find_hyperparameters()