import random
import shutil
from pathlib import Path


CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "del",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "nothing",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "space",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
SEED = 42
TEST_RATIO = 0.2

SRC_DIR = Path("data/cropped")
TRAIN_DIR = Path("data/curated/train")
TEST_DIR = Path("data/curated/test")


def split_dataset() -> None:
    print(
        f"Splitting {SRC_DIR} into train/test "
        f"({1 - TEST_RATIO:.0%}/{TEST_RATIO:.0%})..."
    )

    if not SRC_DIR.exists():
        print(f"Error: source directory {SRC_DIR} not found.")
        print("Run the cropping step first: python scripts/dataset/crop_data.py")
        return

    if TRAIN_DIR.exists():
        print(f"Cleaning existing curated train directory at {TRAIN_DIR}...")
        shutil.rmtree(TRAIN_DIR)
    if TEST_DIR.exists():
        print(f"Cleaning existing curated test directory at {TEST_DIR}...")
        shutil.rmtree(TEST_DIR)

    rng = random.Random(SEED)
    total_train = 0
    total_test = 0

    for sign in CLASSES:
        src_class_dir = SRC_DIR / sign
        if not src_class_dir.exists():
            print(f"  Warning: class {sign} not found, skipping.")
            continue

        images = sorted(src_class_dir.glob("*.jpg"))
        rng.shuffle(images)

        split_idx = int(len(images) * (1 - TEST_RATIO))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        train_dst = TRAIN_DIR / sign
        test_dst = TEST_DIR / sign
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
    print(f"Output: {TRAIN_DIR} and {TEST_DIR}")


def main() -> None:
    split_dataset()


if __name__ == "__main__":
    main()
