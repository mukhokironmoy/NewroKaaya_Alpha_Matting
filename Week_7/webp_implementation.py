from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# --- instrumentation: keep at top ---
import sys, os, faulthandler
faulthandler.enable()
print(">>> Script import start", flush=True)
print(f">>> CWD={os.getcwd()}  ARGV={sys.argv}", flush=True)
# ------------------------------------


# Defaults (can be overridden by CLI)
MIN_DET_CONF = 0.10
MODEL_COMPLEXITY = 2
USE_WEBP = False
SAVE_COMPARISONS = True
ADD_SEPARATOR_LINE = True
SEPARATOR_WIDTH = 6
SEPARATOR_COLOR = (255, 255, 255)  # BGR
JPEG_QUAL = 60
WEBP_QUAL = 65
COMPARE_JPEG_QUAL = 85
INPUT_FOLDER = Path(r"C:\DATA\Internships\Newro Kaaya\PoiseVideos\PoiseVideos\4715_Ganapathy testing _202_20250311155007\4715_Ganapathy testing _202_20250311155007")

def get_crop_box(landmarks, image_shape):
    h, w = image_shape[:2]
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]
    x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
    y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)
    return x_min, y_min, x_max, y_max

def save_side_by_side(original_bgr: np.ndarray,
                      processed_bgr: np.ndarray,
                      out_path: Path,
                      add_separator: bool = True,
                      separator_width: int = 6,
                      separator_color=(255, 255, 255),
                      compare_jpeg_quality: int = 85):
    h1, w1 = original_bgr.shape[:2]
    h2, w2 = processed_bgr.shape[:2]
    if h2 <= 0 or w2 <= 0:
        print(f"⚠️ Processed image is empty, skipping comparison save: {out_path}")
        return
    new_w2 = max(1, int(w2 * (h1 / h2)))
    resized_processed = cv2.resize(processed_bgr, (new_w2, h1), interpolation=cv2.INTER_AREA)
    if add_separator:
        sep = np.full((h1, separator_width, 3), separator_color, dtype=np.uint8)
        combined = np.hstack([original_bgr, sep, resized_processed])
    else:
        combined = np.hstack([original_bgr, resized_processed])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), compare_jpeg_quality])

def process_image(image_path: Path,
                  output_folder: Path,
                  pose: "mp.solutions.pose.Pose",
                  use_webp: bool,
                  save_comparison: bool,
                  comparison_folder: Path | None,
                  add_separator: bool,
                  min_det_conf: float):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ Could not read {image_path}")
        return
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks:
        print(f"Skipping {image_path.name} — no person detected (conf={min_det_conf}).")
        return

    x_min, y_min, x_max, y_max = get_crop_box(results.pose_landmarks.landmark, image.shape)
    cropped = image[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        print(f"⚠️ Empty crop for {image_path.name}, skipping.")
        return

    filename = image_path.stem
    ext = "webp" if use_webp else "jpg"
    output_path = output_folder / f"{filename}.{ext}"

    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    if use_webp:
        pil_img.save(output_path, format="WEBP", quality=WEBP_QUAL)
    else:
        pil_img.save(output_path, format="JPEG", quality=JPEG_QUAL, optimize=True)
    print(f"✅ Saved: {output_path}")

    if save_comparison and comparison_folder is not None:
        comparison_path = comparison_folder / f"{filename}_compare.jpg"
        save_side_by_side(
            original_bgr=image,
            processed_bgr=cropped,
            out_path=comparison_path,
            add_separator=add_separator,
            separator_width=SEPARATOR_WIDTH,
            separator_color=SEPARATOR_COLOR,
            compare_jpeg_quality=COMPARE_JPEG_QUAL
        )
        print(f"🖼️  Comparison: {comparison_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Crop around detected person and compress images.")
    p.add_argument("--folder", type=Path, help="Input folder containing images")
    p.add_argument("--glob", default="*.jpg", help="Glob for images (e.g., *.jpg, *.jpeg, *.png)")
    p.add_argument("--webp", dest="use_webp", action="store_true", help="Save outputs as WebP")
    p.add_argument("--jpeg", dest="use_webp", action="store_false", help="Save outputs as JPEG")
    p.set_defaults(use_webp=USE_WEBP)

    p.add_argument("--compare", dest="save_comp", action="store_true", help="Save side-by-side comparisons")
    p.add_argument("--no-compare", dest="save_comp", action="store_false", help="Do not save comparisons")
    p.set_defaults(save_comp=SAVE_COMPARISONS)

    p.add_argument("--conf", type=float, default=MIN_DET_CONF, help="min_detection_confidence (0.0–1.0)")
    p.add_argument("--complexity", type=int, default=MODEL_COMPLEXITY, choices=[0,1,2], help="Model complexity")
    return p.parse_args()

# if __name__ == "__main__":
    # args = parse_args()

    # # folder = args.folder
    # # if folder is None:
    # #     # Fallback to interactive input if not provided
    # #     try:
    # #         folder = Path(input("Path to folder:\n> ").strip())
    # #     except EOFError:
    # #         print("❌ No folder provided. Use --folder PATH")
    # #         raise SystemExit(1)

    # # if not folder.is_dir():
    # #     print(f"❌ Not a valid folder: {folder}")
    # #     raise SystemExit(1)

    # folder = INPUT_FOLDER

    # output_folder = folder / "cropped_compressed"
    # comparison_folder = folder / "comparisons" if args.save_comp else None

    # output_folder.mkdir(exist_ok=True)
    # if comparison_folder:
    #     comparison_folder.mkdir(exist_ok=True)

    # print(f"📂 Input: {folder}")
    # print(f"📄 Pattern: {args.glob}")
    # print(f"📤 Output: {output_folder}")
    # if comparison_folder:
    #     print(f"🖼️ Comparisons: {comparison_folder}")
    # print(f"⚙️  Options -> WEBP:{args.use_webp}  COMPARE:{args.save_comp}  CONF:{args.conf}  COMPLEXITY:{args.complexity}")

    # mp_pose = mp.solutions.pose
    # with mp_pose.Pose(
    #     static_image_mode=True,
    #     model_complexity=args.complexity,
    #     min_detection_confidence=args.conf
    # ) as pose:
    #     imgs = sorted(folder.glob(args.glob))
    #     if not imgs:
    #         print("⚠️ No images matched the pattern.")
    #     for file_path in imgs:
    #         process_image(
    #             image_path=file_path,
    #             output_folder=output_folder,
    #             pose=pose,
    #             use_webp=args.use_webp,
    #             save_comparison=args.save_comp,
    #             comparison_folder=comparison_folder,
    #             add_separator=ADD_SEPARATOR_LINE,
    #             min_det_conf=args.conf
    #         )
    # print("🎯 Done.")


if __name__ == "__main__":
    print(">>> __main__ reached", flush=True)

    # 1) Choose folder (use your hardcoded INPUT_FOLDER for now)
    folder = INPUT_FOLDER
    print(f">>> Using folder: {folder}", flush=True)
    if not folder.is_dir():
        print(f"❌ Not a valid folder: {folder}", flush=True)
        raise SystemExit(1)

    # 2) Find images (handle .jpg/.JPG/.jpeg; non-recursive)
    imgs = sorted(list(folder.glob("*.jp*g")))
    print(f">>> Found {len(imgs)} images", flush=True)
    if not imgs:
        print("⚠️ No images matched the pattern *.jp*g", flush=True)
        raise SystemExit(0)

    # 3) Prep output folders
    output_folder = folder / "cropped_compressed"
    comparison_folder = folder / "comparisons" if SAVE_COMPARISONS else None
    output_folder.mkdir(exist_ok=True)
    if comparison_folder:
        comparison_folder.mkdir(exist_ok=True)
    print(f">>> Output: {output_folder}", flush=True)
    if comparison_folder:
        print(f">>> Comparisons: {comparison_folder}", flush=True)

    # 4) Run MediaPipe once and process all
    mp_pose = mp.solutions.pose
    print(f">>> Pose config -> complexity={MODEL_COMPLEXITY}  conf={MIN_DET_CONF}", flush=True)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF
    ) as pose:
        for i, file_path in enumerate(imgs, 1):
            print(f"[{i}/{len(imgs)}] Processing {file_path.name}", flush=True)
            process_image(
                image_path=file_path,
                output_folder=output_folder,
                pose=pose,
                use_webp=USE_WEBP,
                save_comparison=SAVE_COMPARISONS,
                comparison_folder=comparison_folder,
                add_separator=ADD_SEPARATOR_LINE,
                min_det_conf=MIN_DET_CONF
            )

    print("🎯 Done.", flush=True)
