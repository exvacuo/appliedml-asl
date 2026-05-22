from collections import defaultdict
import shutil
from pathlib import Path
from imagededup.methods import PHash

CLASSES = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"]
MAX_REMOVALS_PER_CLASS = 1800
DATA_RAW_DIR = Path("data/raw/train/asl_alphabet_train")
OUTPUT_DIR = Path("data/deduplicated/train")

def find_near_duplicates(phasher: PHash, image_dir: str):
    """Find near-duplicate images in a directory using perceptual hashing."""
    phasher = PHash()
    total_to_remove = set()

    for sign in CLASSES:
        image_dir = DATA_RAW_DIR / sign
        encodings = phasher.encode_images(image_dir)
        duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=2)

        seen = set()
        to_remove = set()
        for image, matches in duplicates.items():
            for match in matches:
                pair = frozenset([image, match])
                if pair not in seen:
                    seen.add(pair)
                    to_remove.add(match) 
        total_to_remove.update(to_remove)
    return total_to_remove



def group_by_class(filenames: set) -> dict[str, list[str]]:
    """Group filenames by their class prefix """
    sorted_classes = sorted(CLASSES, key=len, reverse=True)

    grouped = defaultdict(list)
    for fname in filenames:
        for cls in sorted_classes:
            if fname.startswith(cls):
                grouped[cls].append(fname)
                break
    return grouped


def cap_removals(grouped: dict[str, list[str]], cap: int) -> set[str]:
    capped = set()
    for _, files in grouped.items():
        capped.update(files[:cap])
    return capped


def copy_deduplicated_dataset(to_remove: set[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for sign in CLASSES:
        src_class_dir = DATA_RAW_DIR / sign
        dst_class_dir = output_dir / sign
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(src_class_dir.glob("*.jpg"))
        for img_path in images:
            if not img_path.name in to_remove:
                shutil.copy2(img_path, dst_class_dir / img_path.name)
    print(f"Output: {output_dir}")



def main():
    total_to_remove = find_near_duplicates(PHash(), DATA_RAW_DIR)
    print(len(total_to_remove))

    grouped = group_by_class(total_to_remove)

    to_remove_capped = cap_removals(grouped, MAX_REMOVALS_PER_CLASS)

    copy_deduplicated_dataset(to_remove_capped, OUTPUT_DIR)


if __name__ == "__main__":
    main()
