import random
import shutil
from pathlib import Path

from imagededup.methods import PHash
from matplotlib import image

from asl_detector.constants import BALANCE_THRESHOLD_EXCLUDED_CLASSES, CLASSES, DATA_DEDUPLICATED_DIR, DATA_RAW_TRAIN_DIR, OVERSAMPLE_UNDER_TARGET_CLASSES, SEED


def find_near_duplicates(phasher: PHash, image_dir: Path) -> set[str]:
    """Find near-duplicate images in each class directory using perceptual hashing."""
    total_to_remove = set()

    for sign in CLASSES:
        sign_dir = image_dir / sign
        if not sign_dir.exists():
            continue

        encodings = phasher.encode_images(str(sign_dir))
        duplicates = phasher.find_duplicates(
            encoding_map=encodings,
            max_distance_threshold=2,
        )

        seen = set()
        to_remove = set()
        for image in sorted(duplicates.keys()):
            for match in sorted(duplicates[image]):
                pair = frozenset([image, match])
                if pair not in seen:
                    seen.add(pair)
                    to_remove.add(match)
        total_to_remove.update(to_remove)

    return total_to_remove


def copy_downsampled_dataset(to_remove: set[str], output_dir: Path) -> None:
    """Copy a balanced dataset, using the lowest non-excluded class as target."""
    if output_dir.exists():
        print(f"Cleaning existing deduplicated directory at {output_dir}...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_images = {}
    for sign in CLASSES:
        src_class_dir = DATA_RAW_TRAIN_DIR / sign
        if not src_class_dir.exists():
            continue

        images = sorted(src_class_dir.glob("*.jpg"))
        candidate_images = [img for img in images if img.name not in to_remove]

        class_to_images[sign] = candidate_images
        print(f"Class {sign}: {len(images)} total -> {len(candidate_images)} candidates")

    if not class_to_images:
        print("No images found.")
        return

    threshold_candidates = [
        len(images)
        for sign, images in class_to_images.items()
        if sign not in BALANCE_THRESHOLD_EXCLUDED_CLASSES
    ]
    if not threshold_candidates:
        print("No classes available to calculate a balanced target count.")
        return

    target_count = min(threshold_candidates)
    print(
        "\nDownsampling all classes to exactly "
        f"{target_count} images using the lowest non-excluded class..."
    )

    rng = random.Random(SEED)
    total_copied = 0
    for sign, candidate_images in class_to_images.items():
        below_target = len(candidate_images) < target_count
        can_oversample = sign in OVERSAMPLE_UNDER_TARGET_CLASSES
        if below_target and not can_oversample:
            raise ValueError(
                f"Class {sign} only has {len(candidate_images)} candidate images, "
                f"but the balanced target is {target_count}."
            )

        dst_class_dir = output_dir / sign
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        if len(candidate_images) >= target_count:
            sampled_images = rng.sample(candidate_images, target_count)
        else:
            print(
                f"Class {sign} is below target with {len(candidate_images)} images; "
                "oversampling clean candidates with replacement."
            )
            sampled_images = list(candidate_images)
            sampled_images.extend(
                rng.choices(candidate_images, k=target_count - len(candidate_images))
            )

        for index, img_path in enumerate(sampled_images):
            if index < len(candidate_images):
                dst_name = img_path.name
            else:
                dst_name = f"{img_path.stem}_oversampled_{index:04d}{img_path.suffix}"

            shutil.copy2(img_path, dst_class_dir / dst_name)

        total_copied += len(sampled_images)

    print(f"Successfully copied {total_copied} downsampled images to {output_dir}")


def deduplicate_dataset() -> None:
    print("Finding near-duplicates using PHash...")
    total_to_remove = find_near_duplicates(PHash(), DATA_RAW_TRAIN_DIR)
    print(f"Found {len(total_to_remove)} near-duplicates across the dataset.")

    copy_downsampled_dataset(total_to_remove, DATA_DEDUPLICATED_DIR)


def main() -> None:
    deduplicate_dataset()


if __name__ == "__main__":
    main()
