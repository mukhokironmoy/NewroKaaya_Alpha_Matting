from pathlib import Path
from PIL import Image
from clean_dir import clean_directory

# ================= CONFIG =================
INPUT_FOLDER = Path(r"output\convex_hull_output\results")   # change as needed
OUTPUT_FOLDER = Path(r"output\downquality_and_downscale_outputs") # change as needed
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
clean_directory(OUTPUT_FOLDER)

JPEG_QUALITY = 10     # JPEG quality (1–100)
WEBP_QUALITY = 10     # WEBP quality (0–100, lower = smaller, lossy)
SCALE_FACTOR = 1    # e.g. 0.5 = half size, 2.0 = double size
# ===========================================


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
            ext = img_path.suffix.lower().lstrip(".")  # e.g., "jpg", "webp"
            if ext not in ["jpg", "jpeg", "webp"]:
                print(f"⚠️ Skipped {img_path.name} (unsupported format)")
                continue

            with Image.open(img_path) as img:
                # Downscale
                img = downscale_image(img, SCALE_FACTOR)

                # Output path (keep same extension as input)
                out_path = OUTPUT_FOLDER / (img_path.stem + f".{ext}")

                # Save with appropriate quality
                if ext in ["jpg", "jpeg"]:
                    img.save(out_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
                elif ext == "webp":
                    img.save(out_path, "WEBP", quality=WEBP_QUALITY, optimize=True)

                print(f"✅ Processed: {img_path.name} -> {out_path.name}")

        except Exception as e:
            print(f"⚠️ Skipped {img_path.name}: {e}")


if __name__ == "__main__":
    process_images()
