from pathlib import Path

import cv2

from asl_detector.constants import DATA_CROPPED_DIR, DATA_DEDUPLICATED_DIR, PIXELS_TO_CROP


def crop_image(src_path: Path, dst_path: Path, pixels: int = PIXELS_TO_CROP) -> bool:
    img = cv2.imread(str(src_path))
    if img is None:
        return False

    cropped = img[pixels:-pixels, pixels:-pixels]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), cropped)
    return True


def crop_dataset() -> None:
    print("Starting image cropping step...")

    print(f"Cropping images from {DATA_DEDUPLICATED_DIR} to {DATA_CROPPED_DIR}...")
    total_cropped = 0
    for src_class_dir in sorted(path for path in DATA_DEDUPLICATED_DIR.iterdir() if path.is_dir()):
        dst_class_dir = DATA_CROPPED_DIR / src_class_dir.name
        images = sorted(src_class_dir.glob("*.jpg"))

        class_cropped = 0
        for img_path in images:
            if crop_image(img_path, dst_class_dir / img_path.name):
                class_cropped += 1

        total_cropped += class_cropped
        print(f"  Class {src_class_dir.name}: cropped {class_cropped} images.")

    print(f"Total images cropped: {total_cropped}")


def main() -> None:
    crop_dataset()


if __name__ == "__main__":
    main()
