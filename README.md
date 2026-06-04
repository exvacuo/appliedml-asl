# ASL MobileNetV2 predictor readme
Welcome to our repo! This file contains instructions for setting up dependencies and venv for the asl-detector project (Applied ML).

# Setting up the virtual environment
We use uv to manage dependencies in our repository. This project targets Python 3.11. The same `uv.lock` supports macOS and Windows, but each machine should create its own local `.venv`.

In order to install the required dependencies and set up the virtual environment, run:
```bash
uv sync
```

# Deploying the model
Our final model is yet in its training phase, as we are running a wide search space and following a two-phase training (as specified in our proposal) to obtain the best performance possible.

The current weights come from the checkpoint that has gotten the best performance so far. Please note that it does not represent the final expected performance in terms of overfit, accuracy, and inference times (as we have not yet completed the training nor the quantization). The current implementation should be taken as a demonstration of how our API and repo structure is managed.

To deploy the model use the following command:
```bash
uv run uvicorn main:app
```

It will download the latest weights from our Hugging Face repo, and will start the API (default port: 8000). The API documentation, once the API is running, can be found at http://localhost:8000/docs.

# Utilities
Should model training and performance be assessed, use the following helper scripts to check our implementation.
## Generating the dataset
The original dataset comes from Kaggle: [asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data). However, this dataset contains several issues that we had to address via the usage of the scripts, which can be found at `scripts/dataset`.

Before generating the dataset, configure Kaggle credentials. The simplest option is:

```bash
uv run kaggle auth login
```

Alternatively, generate a Kaggle API token and follow Kaggle's instructions for storing it locally.

The complete dataset curation pipeline is encapsulated in `scripts/dataset/craft_dataset.py`. It will follow this pipeline: download from Kaggle → decompress/rename → remove duplicates → crop blue frames → split. It can be executed with (it will take some time):

```bash
uv run scripts/dataset/craft_dataset.py
```

The final dataset splits will be stored to `data/curated`, with 80% for training and 20% for testing. The exact number of images per label can change after deduplication and cropping; the script prints the final counts.

# Training the models
In order to train the models, it is necessary that the dataset has been downloaded and processed. Refer to the previous section for guidance on dataset generation. Once that is done, you can train the baseline (KNN) model using:
```bash
uv run python -m asl_detector.models.knn
```

The complete hyperparameter search space can be searched for the mobilenet final model with:
```bash
uv run python -m asl_detector.models.train.train_mobilenet
```
