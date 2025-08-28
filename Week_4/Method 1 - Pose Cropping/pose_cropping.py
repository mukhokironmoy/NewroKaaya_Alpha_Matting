import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from pathlib import Path

# === CONFIG ===
OUTPUT_FOLDER = r"Method 1 - Pose Cropping\pose_cropping_out"
MARGIN_CM = 2  # Add ~2cm margin around pose
DPI = 96       # Pixels per inch, to estimate pixels per cm
CM_TO_PX = int(MARGIN_CM * DPI / 2.54)  # Convert cm to pixels

# === INIT ===
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5)
frame_id = 0

# === START CAMERA ===
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit...")

while True:
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if results.pose_landmarks:
        # Get all landmark coordinates
        h, w, _ = frame.shape
        x_vals = []
        y_vals = []

        for lm in results.pose_landmarks.landmark:
            if lm.visibility > 0.5:  # Only use clearly visible points
                x_vals.append(int(lm.x * w))
                y_vals.append(int(lm.y * h))

        if x_vals and y_vals:
            # Bounding box + margin
            x_min = max(min(x_vals) - CM_TO_PX, 0)
            x_max = min(max(x_vals) + CM_TO_PX, w)
            y_min = max(min(y_vals) - CM_TO_PX, 0)
            y_max = min(max(y_vals) + CM_TO_PX, h)

            # Crop the frame
            cropped = frame[y_min:y_max, x_min:x_max]

            # Display cropped
            cv2.imshow("Live Pose Crop", cropped)

            # Save the frame
            output_path = Path(OUTPUT_FOLDER) / f"frame_{frame_id:04d}.jpg"
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            pil_img.save(output_path, format="JPEG", optimize=True, quality=95)
            print(f"💾 Saved: {output_path.name}")
            frame_id += 1
        else:
            cv2.imshow("Live Pose Crop", frame)  # fallback
    else:
        cv2.imshow("Live Pose Crop", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Real-time session ended.")
