from pathlib import Path
from PIL import Image
import shutil
from clean_dir import clean_directory

# === CONFIG (Global variables) ===
INPUT_DIR = Path(r"output\convex_hull_output\results")   # Input images
OUTPUT_DIR = Path(r"output\grayscale_outputs")           # Grayscale outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
clean_directory(OUTPUT_DIR)

# Supported formats (add more if needed)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "tiff", "bmp"]

# Quality settings (independent for each format)
JPEG_QUALITY = 80   # 0–100
WEBP_QUALITY = 30   # 0–100 (lower → smaller files, lossy)
PNG_COMPRESSION = 9 # 0–9 (higher = smaller but slower)

def save_with_quality(img: Image.Image, output_path: Path):
    """Save image with correct quality/compression settings based on format."""
    ext = output_path.suffix.lower().lstrip(".")
    save_params = {}

    if ext in ["jpg", "jpeg"]:
        save_params["quality"] = JPEG_QUALITY
        save_params["optimize"] = True
    elif ext == "webp":
        save_params["quality"] = WEBP_QUALITY
    elif ext == "png":
        save_params["compress_level"] = PNG_COMPRESSION

    img.save(output_path, format=img.format, **save_params)

def convert_to_grayscale(image_path: Path, output_path: Path):
    """Convert image to grayscale and save in same format with quality settings."""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")  # Convert to grayscale

            # Ensure output has the same extension
            output_path = output_path.with_suffix(image_path.suffix)

            save_with_quality(gray, output_path)
            print(f"✔ Saved: {output_path}")
    except Exception as e:
        print(f"⚠️ Skipped {image_path.name}: {e}")

def process_folder(input_dir: Path, output_dir: Path):
    """Process all supported images in folder and save to output_dir."""
    if output_dir.exists():
        shutil.rmtree(output_dir)  # Clean output folder each run
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.rglob("*"):
        if file.suffix.lower().lstrip(".") in SUPPORTED_FORMATS:
            relative_path = file.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            convert_to_grayscale(file, output_path)

if __name__ == "__main__":
    if not INPUT_DIR.exists():
        print(f"❌ Input folder does not exist: {INPUT_DIR}")
    else:
        process_folder(INPUT_DIR, OUTPUT_DIR)
        print(f"✅ All images processed. Grayscale images saved to: {OUTPUT_DIR}")
