from pathlib import Path
from PIL import Image

# === Defaults for standalone ===
INPUT_DIR = Path("test_sample")
OUTPUT_FORMAT = "jpg"
JPEG_QUALITY = 100
WEBP_QUALITY = 100

def process_image(img: Image.Image, filename: str,
                  output_format: str = OUTPUT_FORMAT,
                  jpeg_quality: int = JPEG_QUALITY,
                  webp_quality: int = WEBP_QUALITY):
    """Convert image to given format."""
    ext = output_format.lower()
    out_name = Path(filename).with_suffix(f".{ext}")
    if ext in ["jpg", "jpeg"]:
        return img.convert("RGB"), out_name.name, "JPEG", {"quality": jpeg_quality}
    elif ext == "webp":
        return img.convert("RGB"), out_name.name, "WEBP", {"quality": webp_quality}
    return img.convert("RGB"), out_name.name, ext.upper(), {}

def process_file(img_path: Path,
                 output_format: str = OUTPUT_FORMAT,
                 jpeg_quality: int = JPEG_QUALITY,
                 webp_quality: int = WEBP_QUALITY):
    with Image.open(img_path) as img:
        img, new_name, fmt, params = process_image(img, img_path.name,
                                                   output_format, jpeg_quality, webp_quality)
        out_path = img_path.with_suffix(f".{output_format}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, fmt, **params)
        if out_path != img_path:
            img_path.unlink()
        print(f"✅ Converted {img_path.name} → {out_path.name}")

def main():
    for img_path in INPUT_DIR.glob("*.*"):
        process_file(img_path)

if __name__ == "__main__":
    main()
