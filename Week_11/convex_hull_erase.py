import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from clean_dir import clean_directory

# === CONFIG (Global variables you can tweak) ===
INPUT_DIR = Path("test_sample")    # Folder containing input jpgs
OUTPUT_DIR = Path(r"output\convex_hull_output\results")  # Background-removed images
DEBUG_DIR = Path(r"output\convex_hull_output\debug")    # Images with landmarks + hull
PADDING = 60                        # Padding around convex hull (in pixels)

# Supported formats
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "tiff"]

# Select one output format
OUTPUT_FORMAT = "webp"              # "jpg", "png", "webp", "tiff"

# Quality settings (independent for each format)
JPEG_QUALITY = 80                   # 0–100
WEBP_QUALITY = 50                   # 0–100 (lower → smaller files, lossy)
PNG_COMPRESSION = 9                 # 0–9 (higher = smaller but slower)

# Create output folders if they don’t exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
clean_directory(OUTPUT_DIR)
clean_directory(DEBUG_DIR)

# Mediapipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)


def get_body_mask(image, landmarks, padding=PADDING):
    h, w, _ = image.shape

    # Collect landmark points
    points = []
    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])
    points = np.array(points)

    # Convex hull
    hull = cv2.convexHull(points)

    # Add padding by dilating hull mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = np.ones((padding, padding), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, hull, points


def save_image(path: Path, image: np.ndarray):
    """Helper: Save image in chosen format with correct quality params"""
    ext = OUTPUT_FORMAT.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Choose from {SUPPORTED_FORMATS}")

    # Force correct extension
    path = path.with_suffix(f".{ext}")

    if ext in ["jpg", "jpeg"]:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    elif ext == "webp":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])
    elif ext == "png":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    else:  # tiff or others
        cv2.imwrite(str(path), image)

    return path


def process_image(img_path, out_path, debug_path):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"⚠️ Could not read {img_path}")
        return

    # Run pose detection
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        print(f"⚠️ No person detected in {img_path}")
        black_bg = np.zeros_like(image)
        save_image(out_path, black_bg)
        save_image(debug_path, black_bg)
        return

    # Get mask + convex hull + points
    mask, hull, points = get_body_mask(image, results.pose_landmarks)

    # === Background removed result ===
    result = np.zeros_like(image)
    result[mask == 255] = image[mask == 255]
    save_image(out_path, result)

    # === Debug image with landmarks + hull ===
    debug_img = image.copy()
    mp_drawing.draw_landmarks(
        debug_img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
    )
    cv2.polylines(debug_img, [hull], isClosed=True, color=(255, 0, 0), thickness=2)
    save_image(debug_path, debug_img)


def main():
    for img_path in INPUT_DIR.glob("*.jpg"):
        base_name = img_path.stem
        out_path = OUTPUT_DIR / base_name
        debug_path = DEBUG_DIR / base_name
        process_image(img_path, out_path, debug_path)
        print(f"✅ Processed: {img_path.name} → {OUTPUT_FORMAT.upper()}")


if __name__ == "__main__":
    main()
