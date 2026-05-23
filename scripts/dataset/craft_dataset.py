from crop_data import crop_dataset
from deduplicate_data import deduplicate_dataset
from download_data import download_dataset
from split_data import split_dataset


def main() -> None:
    download_dataset()
    deduplicate_dataset()
    crop_dataset()
    split_dataset()
    print("Completed!")


if __name__ == "__main__":
    main()
