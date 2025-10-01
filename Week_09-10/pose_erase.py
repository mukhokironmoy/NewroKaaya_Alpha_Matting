print("Start")
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import os
from typing import Union   # 👈 for backward compatibility

print("Imports Done")

# === HELPER: Clean directory safely ===
def clean_directory(path: Union[str, Path]):   # 👈 fixed type hint
    """Delete all files in a directory. Create dir if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        return
    for item in p.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"⚠️ Could not delete {item}: {e}")

# === CONFIG ===
OUTPUT_FOLDER = r"test_sample"
print("Output folder set")

clean_directory(OUTPUT_FOLDER)  # safe now

MARGIN_CM = 1
DPI = 96  # Dots per inch, used for cm-to-px conversion
CM_TO_PX = int(MARGIN_CM * DPI / 2.54)
print("Variables set")

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Create output folder (ensure exists)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not access webcam. Is it connected?")

frame_id = 0

print("🎥 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Exiting.")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    # Create a blank black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if result.pose_landmarks:
        # Draw filled circles at landmarks
        for lm in result.pose_landmarks.landmark:
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(mask, (cx, cy), CM_TO_PX, 255, -1)

        # Connect joints with lines
        for p1, p2 in mp_pose.POSE_CONNECTIONS:
            pt1 = result.pose_landmarks.landmark[p1]
            pt2 = result.pose_landmarks.landmark[p2]
            if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(mask, (x1, y1), (x2, y2), 255, CM_TO_PX)

        # Smooth + dilate the mask
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Convert mask to 0–1 alpha channel
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0, 1)

        # Apply alpha to frame
        b, g, r = cv2.split(frame)
        rgba = cv2.merge((b, g, r, (alpha * 255).astype(np.uint8)))

        # Save as PNG with alpha
        output_path = Path(OUTPUT_FOLDER) / f"frame_{frame_id:04d}.png"
        pil_img = Image.fromarray(cv2.cvtColor(rgba.astype(np.uint8), cv2.COLOR_BGRA2RGBA))
        pil_img.save(output_path)
        print(f"💾 Saved: {output_path.name}")
        frame_id += 1

        # Create preview with white background
        alpha_3c = np.stack([alpha] * 3, axis=-1)
        foreground = frame.astype(np.float32) * alpha_3c
        white_bg = np.ones_like(frame, dtype=np.float32) * 255
        preview = foreground + white_bg * (1 - alpha_3c)
        preview = preview.astype(np.uint8)

        cv2.imshow("Pose Alpha Matte", preview)

    else:
        cv2.imshow("Pose Alpha Matte", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
