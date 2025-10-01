import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from PIL import Image
from clean_dir import clean_directory

# === Defaults for standalone ===
INPUT_DIR = Path("test_sample")
OUTPUT_DIR = Path("output/convex_hull/results")
DEBUG_DIR = Path("output/convex_hull/debug")
PADDING = 60
SAVE_DEBUG = True
CROP_TO_HULL = False
JPEG_QUALITY, WEBP_QUALITY = 100, 100

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

def save_image(path: Path, image: np.ndarray,
               jpeg_quality=80, webp_quality=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        img_pil.save(path, "JPEG", quality=jpeg_quality, optimize=True)
    elif ext == ".webp":
        img_pil.save(path, "WEBP", quality=webp_quality, method=6)
    else:
        img_pil.save(path)

def crop_to_mask(image: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max(), xs.min():xs.max()]

def process_image(img: np.ndarray, filename: str,
                  crop: bool = CROP_TO_HULL,
                  padding: int = PADDING,
                  jpeg_quality: int = JPEG_QUALITY,
                  webp_quality: int = WEBP_QUALITY,
                  save_debug: bool = SAVE_DEBUG,
                  debug_dir: Path = DEBUG_DIR):
    """Apply convex hull mask and optionally crop."""
    h, w, _ = img.shape
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return np.zeros_like(img), filename, None

    points = np.array([[int(lm.x*w), int(lm.y*h)] for lm in results.pose_landmarks.landmark])
    hull = cv2.convexHull(points)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = np.ones((padding, padding), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    result = np.zeros_like(img)
    result[mask == 255] = img[mask == 255]

    if crop:
        result = crop_to_mask(result, mask)

    debug_img = None
    if save_debug:
        debug_img = img.copy()
        mp_drawing.draw_landmarks(debug_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.polylines(debug_img, [hull], True, (255, 0, 0), 2)
        if crop:
            debug_img = crop_to_mask(debug_img, mask)

    return result, filename, debug_img

def process_file(img_path: Path,
                 crop: bool = CROP_TO_HULL,
                 padding: int = PADDING,
                 jpeg_quality: int = JPEG_QUALITY,
                 webp_quality: int = WEBP_QUALITY,
                 save_debug: bool = SAVE_DEBUG,
                 debug_dir: Path = DEBUG_DIR):
    img = cv2.imread(str(img_path))
    if img is None: return
    res, fname, dbg = process_image(img, img_path.name, crop, padding,
                                    jpeg_quality, webp_quality, save_debug, debug_dir)
    out_path = OUTPUT_DIR / fname
    save_image(out_path, res, jpeg_quality, webp_quality)
    if save_debug and dbg is not None:
        save_image(debug_dir / fname, dbg, jpeg_quality, webp_quality)
    print(f"✅ Processed {img_path.name}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    clean_directory(OUTPUT_DIR)
    clean_directory(DEBUG_DIR)
    for f in INPUT_DIR.glob("*.*"):
        process_file(f)

if __name__ == "__main__":
    main()
