import shutil
import subprocess
import zipfile
from pathlib import Path


DATA_DIR = Path("data/raw")
ZIP_PATH = DATA_DIR / "asl-alphabet.zip"


if shutil.which("kaggle") is None:
    print("You need Kaggle CLI first")
    raise SystemExit(1)

print("Downloading the dataset...")
DATA_DIR.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "grassknoted/asl-alphabet",
        "-p",
        str(DATA_DIR),
        "--force",
    ],
    check=True,
    stdout=subprocess.DEVNULL,
)

print("Decompressing files...")
with zipfile.ZipFile(ZIP_PATH) as dataset:
    dataset.extractall(DATA_DIR)
ZIP_PATH.unlink()

for source, target in (
    (DATA_DIR / "asl_alphabet_train", DATA_DIR / "train"),
    (DATA_DIR / "asl_alphabet_test", DATA_DIR / "test"),
):
    if source.exists():
        if target.exists():
            shutil.rmtree(target)
        source.rename(target)

print("Completed!")
