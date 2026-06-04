import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from scripts.dataset.crop_data import crop_dataset
from scripts.dataset.deduplicate_data import deduplicate_dataset
from scripts.dataset.download_data import download_dataset
from scripts.dataset.split_data import split_dataset


def main() -> None:
    download_dataset()
    deduplicate_dataset()
    crop_dataset()
    split_dataset()
    print("Completed!")


if __name__ == "__main__":
    main()
