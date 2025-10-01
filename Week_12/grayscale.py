from pathlib import Path
from PIL import Image
from clean_dir import clean_directory

# === Defaults for standalone ===
INPUT_DIR = Path(r"output/convex_hull/results")   # fixed path (use raw string or /)
OUTPUT_DIR = Path("output/grayscale")
JPEG_QUALITY, WEBP_QUALITY = 100, 100

def process_image(img: Image.Image, filename: str,
                  jpeg_quality: int = 100,
                  webp_quality: int = 100):
    gray = img.convert("L")
    ext = filename.split(".")[-1].lower()
    if ext in ["jpg", "jpeg"]:
        return gray, filename, "JPEG", {"quality": jpeg_quality}
    elif ext == "webp":
        return gray, filename, "WEBP", {"quality": webp_quality}
    return gray, filename, img.format, {}

def process_file(img_path: Path,
                 jpeg_quality: int = JPEG_QUALITY,
                 webp_quality: int = WEBP_QUALITY):
    with Image.open(img_path) as img:
        gray, fname, fmt, params = process_image(img, img_path.name,
                                                 jpeg_quality, webp_quality)
        out_path = OUTPUT_DIR / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gray.save(out_path, fmt, **params)
        print(f"✅ Grayscaled {img_path.name}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_directory(OUTPUT_DIR)
    for f in INPUT_DIR.glob("*.*"):
        process_file(f)

if __name__ == "__main__":
    main()
