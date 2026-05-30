import random
import shutil
from pathlib import Path

from asl_detector.constants import CLASSES, DATA_CROPPED_DIR, DATA_CURATED_TEST_DIR, DATA_CURATED_TRAIN_DIR, SEED, TEST_RATIO


def split_dataset() -> None:
    print(
        f"Splitting {DATA_CROPPED_DIR} into train/test "
        f"({1 - TEST_RATIO:.0%}/{TEST_RATIO:.0%})..."
    )

    if not DATA_CROPPED_DIR.exists():
        print(f"Error: source directory {DATA_CROPPED_DIR} not found.")
        print("Run the cropping step first: python scripts/dataset/crop_data.py")
        return

    if DATA_CURATED_TRAIN_DIR.exists():
        print(f"Cleaning existing curated train directory at {DATA_CURATED_TRAIN_DIR}...")
        shutil.rmtree(DATA_CURATED_TRAIN_DIR)
    if DATA_CURATED_TEST_DIR.exists():
        print(f"Cleaning existing curated test directory at {DATA_CURATED_TEST_DIR}...")
        shutil.rmtree(DATA_CURATED_TEST_DIR)

    rng = random.Random(SEED)
    total_train = 0
    total_test = 0

    for sign in CLASSES:
        src_class_dir = DATA_CROPPED_DIR / sign
        if not src_class_dir.exists():
            print(f"  Warning: class {sign} not found, skipping.")
            continue

        images = sorted(src_class_dir.glob("*.jpg"))
        rng.shuffle(images)

        split_idx = int(len(images) * (1 - TEST_RATIO))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        train_dst = DATA_CURATED_TRAIN_DIR / sign
        test_dst = DATA_CURATED_TEST_DIR / sign
        train_dst.mkdir(parents=True, exist_ok=True)
        test_dst.mkdir(parents=True, exist_ok=True)

        for img_path in train_images:
            shutil.copy2(img_path, train_dst / img_path.name)
        for img_path in test_images:
            shutil.copy2(img_path, test_dst / img_path.name)

        total_train += len(train_images)
        total_test += len(test_images)
        print(f"  Class {sign}: {len(train_images)} train, {len(test_images)} test")

    print(f"\nTotal: {total_train} train, {total_test} test")
    print(f"Output: {DATA_CURATED_TRAIN_DIR} and {DATA_CURATED_TEST_DIR}")


def main() -> None:
    split_dataset()


if __name__ == "__main__":
    main()
