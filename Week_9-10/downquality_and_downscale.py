from pathlib import Path
from PIL import Image
from clean_dir import clean_directory

# ================= CONFIG =================
INPUT_FOLDER = Path(r"test_sample")   # change as needed
OUTPUT_FOLDER = Path(r"outputs\downquality_and_downscale_outputs") # change as needed
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
clean_directory(OUTPUT_FOLDER)

QUALITY_FACTOR = 40   # JPEG quality (1–100)
SCALE_FACTOR = 0.5    # e.g. 0.5 = half size, 2.0 = double size
# ===========================================


def reduce_quality(img: Image.Image, quality: int) -> Image.Image:
    """
    Save the image with reduced JPEG quality.
    Returns the same image object (not altered in memory).
    """
    return img  # quality is applied only when saving


def downscale_image(img: Image.Image, scale: float) -> Image.Image:
    """
    Downscale the image by a scale factor.
    Example: 0.5 -> half size, 2.0 -> double size.
    """
    w, h = img.size
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def process_images():
    for img_path in INPUT_FOLDER.glob("*.*"):
        try:
            with Image.open(img_path) as img:
                # Downscale
                img = downscale_image(img, SCALE_FACTOR)

                # Output path
                out_path = OUTPUT_FOLDER / (img_path.stem + ".jpg")

                # Save with reduced quality
                img.save(out_path, "JPEG", quality=QUALITY_FACTOR, optimize=True)

                print(f"✅ Processed: {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"⚠️ Skipped {img_path.name}: {e}")


if __name__ == "__main__":
    process_images()
