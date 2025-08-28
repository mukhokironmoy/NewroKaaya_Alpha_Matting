# webp_run.py
from pathlib import Path
import sys, os, faulthandler
faulthandler.enable()

print(">>> import ok", flush=True)

INPUT_FOLDER = Path(r"C:\DATA\Internships\Newro Kaaya\PoiseVideos\PoiseVideos\4715_Ganapathy testing _202_20250311155007\4715_Ganapathy testing _202_20250311155007")

def main():
    print(">>> main() entered", flush=True)
    print("Using folder:", INPUT_FOLDER, flush=True)
    if not INPUT_FOLDER.is_dir():
        print("❌ Not a valid folder", flush=True)
        return
    imgs = sorted(list(INPUT_FOLDER.glob("*.jp*g")))
    print("Found", len(imgs), "images", flush=True)
    for i, p in enumerate(imgs[:3], 1):
        print(f"[{i}] {p.name}", flush=True)
    print(">>> main() exit", flush=True)

if __name__ == "__main__":
    print(">>> __main__ reached", flush=True)
    main()
    print("🎯 Done.", flush=True)
