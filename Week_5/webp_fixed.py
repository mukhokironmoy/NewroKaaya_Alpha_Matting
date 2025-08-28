# webp_fixed.py — debug-friendly, full pipeline (Py 3.9-safe)

# ---- early instrumentation: these must stay at the very top ----
import sys, os, faulthandler
faulthandler.enable()
print("__name__ =", repr(__name__), flush=True)
print(">>> Running from:", os.path.abspath(__file__), flush=True)
print(">>> CWD:", os.getcwd(), "ARGV:", sys.argv, flush=True)
print(">>> pre-main imports ok", flush=True)
# ----------------------------------------------------------------

# ---- config (pure Python; safe at top) ----
# Segmentation + crop settings
SEG_THRESH = 0.05         # threshold on [0..1] mask to keep foreground
CROP_PADDING = 20         # px padding around bbox when DO_CROP is True
DO_CROP = True           # <<< set True to crop around detected person mask

# Output/settings
USE_WEBP = True          # True: .webp with alpha, False: .jpg (white bg fallback)
SAVE_COMPARISONS = True
ADD_SEPARATOR_LINE = True
SEPARATOR_WIDTH = 6
SEPARATOR_COLOR = (255, 255, 255)  # BGR for OpenCV
JPEG_QUAL = 60
WEBP_QUAL = 55            # quality for WebP (lossy), keeps alpha
COMPARE_JPEG_QUAL = 85
RECURSIVE = False         # set True to scan subfolders

# (kept for logs/backward compat; not used by segmentation)
MIN_DET_CONF = 0.01
MODEL_COMPLEXITY = 2

INPUT_FOLDER = r"C:\DATA\Internships\Newro Kaaya\Test_02\test_data"
# ---------------------------------------------------

def main():
    print(">>> main() entered", flush=True)

    # --- import heavy libs INSIDE main with checkpoints ---
    print(".. importing Path", flush=True)
    from pathlib import Path

    print(".. importing typing.Optional", flush=True)
    from typing import Optional, Tuple

    print(".. importing cv2", flush=True)
    import cv2

    print(".. importing numpy", flush=True)
    import numpy as np

    print(".. importing PIL.Image", flush=True)
    from PIL import Image

    print(".. importing mediapipe.selfie_segmentation", flush=True)
    import mediapipe as mp
    mp_ss = mp.solutions.selfie_segmentation

    print("✅ all imports inside main() ok", flush=True)
    # ------------------------------------------------------

    folder = Path(INPUT_FOLDER)
    print(f">>> Using folder: {folder}", flush=True)
    if not folder.is_dir():
        print(f"❌ Not a valid folder: {folder}", flush=True)
        print(f"Exists? {folder.exists()}  Is dir? {folder.is_dir()}", flush=True)
        return

    # --- helpers ---
    def mask_bbox(mask: np.ndarray, thresh: float, pad: int, shape: Tuple[int, int]) -> Optional[Tuple[int,int,int,int]]:
        """Get padded bbox (x0,y0,x1,y1) of foreground in mask; return None if empty."""
        h, w = shape
        ys, xs = np.where(mask > thresh)
        if ys.size == 0 or xs.size == 0:
            return None
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w - 1)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h - 1)
        return x0, y0, x1, y1

    def remove_background_rgba(bgr: np.ndarray, seg_mask: np.ndarray, thresh: float) -> np.ndarray:
        """Return BGRA with alpha from segmentation mask."""
        # seg_mask is float32 [0,1]; build 8-bit alpha
        alpha = np.clip((seg_mask - thresh) / max(1e-6, (1.0 - thresh)), 0.0, 1.0)
        alpha8 = (alpha * 255).astype(np.uint8)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha8
        return bgra

    def composite_on_white(bgra: np.ndarray) -> np.ndarray:
        """Composite BGRA over white background -> BGR (for JPEG/preview)."""
        bgr = bgra[:, :, :3].astype(np.float32)
        a = (bgra[:, :, 3:4].astype(np.float32)) / 255.0
        white = np.full_like(bgr, 255.0)
        out = bgr * a + white * (1.0 - a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def save_side_by_side(original_bgr,
                          processed_preview_bgr,
                          out_path: Path,
                          add_separator: bool = True,
                          separator_width: int = 6,
                          separator_color=(255, 255, 255),
                          compare_jpeg_quality: int = 85):
        h1, w1 = original_bgr.shape[:2]
        h2, w2 = processed_preview_bgr.shape[:2]
        if h2 <= 0 or w2 <= 0:
            print(f"⚠️ Empty processed; skip {out_path}")
            return
        new_w2 = max(1, int(w2 * (h1 / h2)))
        resized = cv2.resize(processed_preview_bgr, (new_w2, h1), interpolation=cv2.INTER_AREA)
        if add_separator:
            sep = np.full((h1, separator_width, 3), separator_color, dtype=np.uint8)
            combined = np.hstack([original_bgr, sep, resized])
        else:
            combined = np.hstack([original_bgr, resized])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), compare_jpeg_quality])

    def process_image(image_path: Path,
                      output_folder: Path,
                      segmenter: "mp_ss.SelfieSegmentation",
                      use_webp: bool,
                      save_comparison: bool,
                      comparison_folder: Optional[Path],
                      add_separator: bool,
                      seg_thresh: float,
                      do_crop: bool):
        import cv2
        import numpy as np
        from PIL import Image

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"⚠️ Could not read {image_path}")
            return

        # ---- segmentation (always) ----
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb)
        seg = results.segmentation_mask  # float32 [0,1], higher = person/fg

        if seg is None:
            print(f"⚠️ No segmentation mask for {image_path.name} — skipping.")
            return

        # Optional crop based on mask bbox
        if do_crop:
            bbox = mask_bbox(seg, seg_thresh, CROP_PADDING, image_bgr.shape[:2])
            if bbox is None:
                print(f"Skip {image_path.name} — no foreground found for crop (thresh={seg_thresh})")
                return
            x0, y0, x1, y1 = bbox
            image_bgr_c = image_bgr[y0:y1+1, x0:x1+1]
            seg_c = seg[y0:y1+1, x0:x1+1]
        else:
            image_bgr_c = image_bgr
            seg_c = seg

        # Remove background -> BGRA
        image_bgra = remove_background_rgba(image_bgr_c, seg_c, seg_thresh)

        # Save: WebP with alpha, or JPEG composited on white
        filename = image_path.stem
        if use_webp:
            out_path = output_folder / f"{filename}.webp"
            # PIL can save WebP with alpha directly
            pil_rgba = Image.fromarray(cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGBA))
            pil_rgba.save(out_path, format="WEBP", quality=WEBP_QUAL)
        else:
            out_path = output_folder / f"{filename}.jpg"
            preview_bgr = composite_on_white(image_bgra)
            pil_rgb = Image.fromarray(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB))
            pil_rgb.save(out_path, format="JPEG", quality=JPEG_QUAL, optimize=True)

        print(f"✅ Saved: {out_path}")

        # Save comparison (JPEG) — show processed preview over white for visibility
        if save_comparison and comparison_folder:
            preview_bgr = composite_on_white(image_bgra)
            comp_path = comparison_folder / f"{filename}_compare.jpg"
            save_side_by_side(
                image_bgr, preview_bgr, comp_path,
                add_separator, SEPARATOR_WIDTH, SEPARATOR_COLOR, COMPARE_JPEG_QUAL
            )
            print(f"🖼️  Comparison: {comp_path}")
    # --------------------------------------------------------------

    # collect images (case-insensitive .jpg/.jpeg). Use rglob if RECURSIVE
    exts = {".jpg", ".jpeg"}
    if RECURSIVE:
        candidates = (p for p in folder.rglob("*") if p.is_file())
    else:
        candidates = (p for p in folder.iterdir() if p.is_file())
    imgs = sorted(p for p in candidates if p.suffix.lower() in exts)

    print(f">>> Found {len(imgs)} images (recursive={RECURSIVE})", flush=True)
    if not imgs:
        try:
            names = [p.name for p in list(folder.iterdir())[:10]]
            print("DEBUG: first files in folder:", names, flush=True)
        except Exception as e:
            print("DEBUG: could not list folder:", e, flush=True)
        return

    out_dir = folder / ("bg_removed" if not DO_CROP else "bg_removed_cropped")
    comp_dir: Optional[Path] = folder / "comparisons" if SAVE_COMPARISONS else None
    out_dir.mkdir(exist_ok=True)
    if comp_dir:
        comp_dir.mkdir(exist_ok=True)

    print(f">>> Segmentation config -> thresh={SEG_THRESH}  do_crop={DO_CROP}", flush=True)
    with mp_ss.SelfieSegmentation(model_selection=1) as segmenter:
        for i, fp in enumerate(imgs, 1):
            print(f"[{i}/{len(imgs)}] {fp.name}", flush=True)
            process_image(fp, out_dir, segmenter, USE_WEBP, SAVE_COMPARISONS, comp_dir,
                          ADD_SEPARATOR_LINE, SEG_THRESH, DO_CROP)

    print(">>> main() exit", flush=True)

# ---- robust footer so you see exactly what happens ----
print(">>> End of file (before main check) reached", flush=True)
if __name__ == "__main__":
    print(">>> __main__ reached", flush=True)
    try:
        main()
    except Exception as e:
        print("❌ Exception in main():", repr(e), flush=True)
        import traceback
        traceback.print_exc()
    print("🎯 Done.", flush=True)
print(">>> End of file (after main check) reached", flush=True)
