from pathlib import Path
from PIL import Image

# === CONFIG ===
INPUT_DIR = Path(r"test_sample")  # <-- set your folder path here

def convert_jpg_to_webp_inplace(folder: Path):
    """Convert all JPG images in the folder to WebP format, replacing originals."""
    for img_path in folder.glob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                # New path with same stem but .webp extension
                webp_path = img_path.with_suffix(".webp")
                
                # Save as WebP
                img.save(webp_path, "WEBP", quality=80)

            # Remove original JPG
            img_path.unlink()

            print(f"✅ Replaced: {img_path.name} -> {webp_path.name}")

        except Exception as e:
            print(f"⚠️ Failed to convert {img_path.name}: {e}")

if __name__ == "__main__":
    convert_jpg_to_webp_inplace(INPUT_DIR)
