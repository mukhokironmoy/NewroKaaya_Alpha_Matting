import random
import shutil
from pathlib import Path
from clean_dir import clean_directory

# === Defaults for standalone ===
INPUT_DIR = Path(r"C:\PoiseVideos\4825_Prasanna K.B_189_20250916173306\4825_Prasanna K.B_189_20250916173306")        # main dataset
OUTPUT_DIR = Path("test_sample")
SAMPLE_SIZE = 100

def create_test_sample(input_dir: Path, output_dir: Path, sample_size: int):
    """Copy a random sample of images from input_dir to output_dir."""
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".webp"}])
    if not images:
        raise ValueError(f"No images found in {input_dir}")
    if sample_size > len(images):
        raise ValueError("Sample size larger than available images")

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_directory(output_dir)

    sampled = random.sample(images, sample_size)
    for img in sampled:
        shutil.copy(img, output_dir / img.name)

    print(f"✅ Copied {sample_size} images to {output_dir}")

if __name__ == "__main__":
    create_test_sample(INPUT_DIR, OUTPUT_DIR, SAMPLE_SIZE)
