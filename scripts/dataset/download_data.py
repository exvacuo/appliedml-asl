import shutil
import subprocess
import zipfile
from pathlib import Path

from asl_detector.constants import DATA_RAW_BASE_DIR, KAGGLE_DATASET, ZIP_PATH


def download_dataset() -> None:
    """Download and extract the ASL alphabet dataset from Kaggle."""
    if shutil.which("kaggle") is None:
        print("You need Kaggle CLI first")
        raise SystemExit(1)

    print("Downloading the dataset...")
    DATA_RAW_BASE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(DATA_RAW_BASE_DIR),
            "--force",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    print("Decompressing files (this might still take a minute on cluster storage)...")
    with zipfile.ZipFile(ZIP_PATH) as zip_file:
        zip_file.extractall(DATA_RAW_BASE_DIR)
    ZIP_PATH.unlink()

    for source, target in (
        (DATA_RAW_BASE_DIR / "asl_alphabet_train", DATA_RAW_BASE_DIR / "train"),
        (DATA_RAW_BASE_DIR / "asl_alphabet_test", DATA_RAW_BASE_DIR / "test"),
    ):
        if source.exists():
            if target.exists():
                shutil.rmtree(target)
            source.rename(target)

    print("Download completed!")


if __name__ == "__main__":
    download_dataset()
