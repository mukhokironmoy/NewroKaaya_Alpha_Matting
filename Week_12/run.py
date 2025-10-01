from pathlib import Path
from PIL import Image
import cv2, numpy as np, shutil, json

# Import helpers
from sample_creator import create_test_sample
from convert_format import process_image as convert_img
from convex_hull import process_image as convex_img
from downquality_and_downscale import process_image as downscale_img
from grayscale import process_image as gray_img

# ===================================================
# CONFIG (two options)
# ===================================================

# Option 1: Paste JSON as a string here
CONFIG_JSON = """
{
    "name": "WebP Grayscale",
    "SAMPLE_INPUT_DIR": "C:/PoiseVideos/4825_Prasanna K.B_189_20250916173306/4825_Prasanna K.B_189_20250916173306",
    "SAMPLE_SIZE": 50,
    "TOGGLE_SAMPLE_CREATOR": true,
    "TOGGLE_CONVERT_FORMAT": true,
    "TOGGLE_CONVEX_HULL": true,
    "TOGGLE_CONVEX_HULL_CROP": false,
    "TOGGLE_GRAYSCALE": true,
    "TOGGLE_DOWNSCALE": false,
    "OUTPUT_FORMAT": "webp",
    "JPEG_QUALITY": 100,
    "WEBP_QUALITY": 100,
    "SCALE_FACTOR": 1,
    "PADDING": 60,
    "SAVE_DEBUG": false
  }
"""

# Option 2: Load from external JSON file (uncomment this)
# CONFIG_PATH = Path("config.json")
# with open(CONFIG_PATH, "r", encoding="utf-8") as f:
#     CONFIG = json.load(f)

# Use Option 1 by default
CONFIG = json.loads(CONFIG_JSON)

# --- Convert certain keys into Path objects ---
if "SAMPLE_INPUT_DIR" in CONFIG:
    CONFIG["SAMPLE_INPUT_DIR"] = Path(CONFIG["SAMPLE_INPUT_DIR"])
if "SAMPLE_OUTPUT_DIR" not in CONFIG:
    CONFIG["SAMPLE_OUTPUT_DIR"] = Path("test_sample")

# Output dirs
OUTPUT_DIR = Path("output")
FINAL_RESULT_DIR = OUTPUT_DIR / "final_results"


def pipeline(cfg: dict):
    FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Sample creation
    if cfg["TOGGLE_SAMPLE_CREATOR"]:
        create_test_sample(cfg["SAMPLE_INPUT_DIR"], cfg["SAMPLE_OUTPUT_DIR"], cfg["SAMPLE_SIZE"])
        source = cfg["SAMPLE_OUTPUT_DIR"]
    else:
        source = cfg["SAMPLE_INPUT_DIR"]

    files = list(source.glob("*.*"))
    if not files:
        print(f"⚠️ No files found in {source}")
        return

    print(f"📂 Found {len(files)} files to process from {source}")

    for f in files:
        try:
            img = Image.open(f)
            fname, fmt, params = f.name, img.format, {}

            # Stage 2: Convert format
            if cfg["TOGGLE_CONVERT_FORMAT"]:
                img, fname, fmt, params = convert_img(
                    img, fname,
                    cfg["OUTPUT_FORMAT"],
                    cfg["JPEG_QUALITY"],
                    cfg["WEBP_QUALITY"]
                )

            # Stage 3: Convex hull
            if cfg["TOGGLE_CONVEX_HULL"]:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv_res, fname, dbg = convex_img(
                    cv_img, fname,
                    crop=cfg["TOGGLE_CONVEX_HULL_CROP"],
                    padding=cfg["PADDING"],
                    jpeg_quality=cfg["JPEG_QUALITY"],
                    webp_quality=cfg["WEBP_QUALITY"],
                    save_debug=cfg["SAVE_DEBUG"],
                    debug_dir=OUTPUT_DIR / "convex_hull/debug"
                )
                img = Image.fromarray(cv2.cvtColor(cv_res, cv2.COLOR_BGR2RGB))

            # Stage 4: Downscale
            if cfg["TOGGLE_DOWNSCALE"]:
                img, fname, fmt, params = downscale_img(
                    img, fname,
                    scale_factor=cfg["SCALE_FACTOR"],
                    jpeg_quality=cfg["JPEG_QUALITY"],
                    webp_quality=cfg["WEBP_QUALITY"]
                )

            # Stage 5: Grayscale
            if cfg["TOGGLE_GRAYSCALE"]:
                img, fname, fmt, params = gray_img(
                    img, fname,
                    jpeg_quality=cfg["JPEG_QUALITY"],
                    webp_quality=cfg["WEBP_QUALITY"]
                )

            # Final save
            out_path = FINAL_RESULT_DIR / fname
            img.save(out_path, fmt, **params)
            print(f"✅ Final saved {fname}")

        except Exception as e:
            print(f"⚠️ Failed {f.name}: {e}")


if __name__ == "__main__":
    if FINAL_RESULT_DIR.exists():
        shutil.rmtree(FINAL_RESULT_DIR)
    FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Running pipeline: {CONFIG['name']}")
    pipeline(CONFIG)
    print("🎉 Pipeline finished!")
