from pathlib import Path
from PIL import Image
from clean_dir import clean_directory

# === Defaults for standalone ===
INPUT_DIR = Path(r"output\grayscale")
OUTPUT_DIR = Path(r"output/downscaled")
SCALE_FACTOR = 1
JPEG_QUALITY, WEBP_QUALITY = 70, 70

def process_image(img: Image.Image, filename: str,
                  scale_factor: float = SCALE_FACTOR,
                  jpeg_quality: int = JPEG_QUALITY,
                  webp_quality: int = WEBP_QUALITY):
    w, h = img.size
    new_size = (int(w*scale_factor), int(h*scale_factor))
    out_img = img.resize(new_size, Image.LANCZOS)
    ext = filename.split(".")[-1].lower()
    if ext in ["jpg","jpeg"]:
        return out_img, filename, "JPEG", {"quality": jpeg_quality}
    elif ext == "webp":
        return out_img, filename, "WEBP", {"quality": webp_quality}
    return out_img, filename, img.format, {}

def process_file(img_path: Path,
                 scale_factor: float = SCALE_FACTOR,
                 jpeg_quality: int = JPEG_QUALITY,
                 webp_quality: int = WEBP_QUALITY):
    with Image.open(img_path) as img:
        out_img, fname, fmt, params = process_image(img, img_path.name,
                                                    scale_factor, jpeg_quality, webp_quality)
        out_path = OUTPUT_DIR / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, fmt, **params)
        print(f"✅ Downscaled {img_path.name}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_directory(OUTPUT_DIR)
    for f in INPUT_DIR.glob("*.*"):
        process_file(f)

if __name__ == "__main__":
    main()
