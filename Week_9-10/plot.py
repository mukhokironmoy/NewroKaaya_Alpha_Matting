import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from clean_dir import clean_directory

# MediaPipe Pose connections (edges between landmark indices)
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

def process_images_with_landmarks(image_dir: Path, mapping_file: Path):
    # --- Load mapping.json ---
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # --- Prepare output folder ---
    output_dir = Path(r"outputs\plot_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_directory(output_dir)

    # --- Collect images ---
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]
    )

    for img_path in images:
        img_name = img_path.name
        img = mpimg.imread(img_path)

        # --- Create figure ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # --- Case 1: Landmark exists ---
        if img_name in mapping:
            img_info = mapping[img_name]
            landmarks = np.array(img_info["landmarks"], dtype=float)
            xs, ys = landmarks[:, 0], landmarks[:, 1]

            # Left: raw skeleton on black background
            axes[0].set_title("Pose Skeleton (original scale)")
            axes[0].set_facecolor("black")
            draw_pose(axes[0], xs, ys, "cyan")
            axes[0].invert_yaxis()
            axes[0].axis("equal")
            axes[0].axis("off")

            # Right: skeleton scaled and overlaid on image
            axes[1].imshow(img)
            axes[1].set_title("Skeleton scaled to image")

            # Scale landmarks to fit compressed image
            h, w = img.shape[:2]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            scale_x = w / (x_max - x_min) if (x_max - x_min) != 0 else 1
            scale_y = h / (y_max - y_min) if (y_max - y_min) != 0 else 1
            scale = min(scale_x, scale_y)

            xs_scaled = (xs - x_min) * scale
            ys_scaled = (ys - y_min) * scale

            draw_pose(axes[1], xs_scaled, ys_scaled, "yellow")
            axes[1].axis("off")

        # --- Case 2: No mapping for this image ---
        else:
            # Left: empty black canvas
            axes[0].set_title("No landmarks available")
            axes[0].set_facecolor("black")
            axes[0].axis("off")

            # Right: plain image
            axes[1].imshow(img)
            axes[1].set_title("Original Image")
            axes[1].axis("off")

        # --- Save result ---
        out_file = output_dir / f"{img_path.stem}_processed.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close(fig)

        print(f"✅ Processed {img_name} -> {out_file}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    image_dir = Path("test_sample")         # directory with images
    mapping_file = Path("mapping.json")  # your mapping.json
    process_images_with_landmarks(image_dir, mapping_file)
