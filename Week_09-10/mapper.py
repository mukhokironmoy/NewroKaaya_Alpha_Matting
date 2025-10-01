import json
from pathlib import Path

def parse_landmark_line(line: str):
    """Convert 'x,y;x,y;...' string into list of (x, y) floats."""
    coords = []
    for pair in line.split(";"):
        pair = pair.strip()
        if pair and not pair.startswith("/"):  # skip empty or trailing /id
            x, y = map(float, pair.split(","))
            coords.append((x, y))
    return coords

def create_mapping(landmark_file: Path, image_dir: Path, output_dir: Path):
    # --- Load landmarks ---
    with open(landmark_file, "r", encoding="utf-8") as f:
        landmark_lines = [line.strip() for line in f if line.strip()]
    landmarks = [parse_landmark_line(line) for line in landmark_lines]

    # --- Collect images ---
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]
    )

    # --- Align lengths ---
    n = min(len(images), len(landmarks))
    images, landmarks = images[:n], landmarks[:n]

    # --- Build mapping (image_path → list of (x, y)) ---
    mapping = {str(img): coords for img, coords in zip(images, landmarks)}

    # --- Prepare output folder ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "mapping.json"

    # --- Save JSON ---
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"✅ Mapping saved to {out_file} ({len(mapping)} pairs)")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    landmark_file = Path(r"PoiseVideos\4715_Ganapathy testing _202_20250311155007\4715_Ganapathy testing _202_20250311155007_body.btr2d")       # your landmark file
    image_dir = Path(r"PoiseVideos\4715_Ganapathy testing _202_20250311155007\4715_Ganapathy testing _202_20250311155007")                  # folder with images
    output_dir = Path(".")              # will be created if missing

    create_mapping(landmark_file, image_dir, output_dir)
