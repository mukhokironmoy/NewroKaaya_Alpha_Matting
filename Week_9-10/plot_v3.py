import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from clean_dir import clean_directory

# MediaPipe Pose connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

def draw_pose(ax, xs, ys, color="red"):
    """Draw landmarks as connected skeleton."""
    ax.scatter(xs, ys, c=color, s=15)
    for i, j in POSE_CONNECTIONS:
        if i < len(xs) and j < len(xs):
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=color, linewidth=2)

def normalize_and_scale(xs, ys, w, h, flip_horizontal=False):
    """Normalize landmarks to [0,1] then scale to image size."""
    # Normalize
    xs_norm = (xs - xs.min()) / (xs.max() - xs.min())
    ys_norm = (ys - ys.min()) / (ys.max() - ys.min())

    # Scale to image
    xs_scaled = xs_norm * w
    ys_scaled = ys_norm * h

    # Optional flip
    if flip_horizontal:
        xs_scaled = w - xs_scaled

    return xs_scaled, ys_scaled

def process_images_with_landmarks(image_dir: Path, mapping_file: Path, flip_horizontal=False):
    # --- Load mapping.json ---
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # --- Prepare output folder ---
    output_dir = Path(r"outputs\plot_v3_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_directory(output_dir)

    # --- Collect images ---
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]
    )

    for img_path in images:
        img_name = img_path.name
        img = mpimg.imread(img_path)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        if img_name in mapping:
            img_info = mapping[img_name]
            landmarks = np.array(img_info["landmarks"], dtype=float)
            xs, ys = landmarks[:, 0], landmarks[:, 1]

            # --- Normalize & scale landmarks to compressed image ---
            h, w = img.shape[:2]
            xs_scaled, ys_scaled = normalize_and_scale(xs, ys, w, h, flip_horizontal)

            # Left: normalized skeleton on black
            axes[0].set_title("Pose Skeleton (normalized)")
            axes[0].set_facecolor("black")
            draw_pose(axes[0], xs_scaled, ys_scaled, "cyan")
            axes[0].invert_yaxis()   # match image coordinate system
            axes[0].axis("equal")
            axes[0].axis("off")

            # Right: overlay on image
            axes[1].imshow(img)
            axes[1].set_title("Skeleton aligned to image")
            draw_pose(axes[1], xs_scaled, ys_scaled, "yellow")
            axes[1].axis("off")

        else:
            # Left: black canvas
            axes[0].set_title("No landmarks available")
            axes[0].set_facecolor("black")
            axes[0].axis("off")

            # Right: plain image
            axes[1].imshow(img)
            axes[1].set_title("Original Image")
            axes[1].axis("off")

        out_file = output_dir / f"{img_path.stem}_processed.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close(fig)

        print(f"✅ Processed {img_name} -> {out_file}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    image_dir = Path("test_sample")
    mapping_file = Path("mapping.json")
    process_images_with_landmarks(image_dir, mapping_file, flip_horizontal=True)
