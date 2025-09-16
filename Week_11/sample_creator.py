import random
import shutil
from pathlib import Path
from clean_dir import clean_directory

def create_test_sample(image_dir: Path, output_dir: Path, sample_size: int):
    # --- Collect all images ---
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]
    )

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    # --- Ensure sample size is valid ---
    if sample_size > len(images):
        raise ValueError(f"Sample size {sample_size} is larger than available images ({len(images)})")

    # --- Pick random sample ---
    sampled_images = random.sample(images, sample_size)

    # --- Create output folder ---
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_directory(output_dir)

    # --- Copy files ---
    for img in sampled_images:
        shutil.copy(img, output_dir / img.name)

    print(f"✅ Copied {sample_size} images to {output_dir}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    image_dir = Path(r"C:\PoiseVideos\4825_Prasanna K.B_189_20250916173306\4825_Prasanna K.B_189_20250916173306")                  # folder with images
    output_dir = Path("test_sample")    # new folder to be created
    sample_size = 100                  # number of images to copy

    create_test_sample(image_dir, output_dir, sample_size)
