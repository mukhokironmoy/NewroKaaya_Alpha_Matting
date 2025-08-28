import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from pathlib import Path
import time

# === CONFIG ===
OUTPUT_FOLDER = r"Method 3 - Real time segmentation and cropping\realtime_out"
CONFIDENCE_THRESHOLD = 0.3
MARGIN = 30
SAVE_DETECTED_ONLY = True  # Only save if person detected

# === INIT ===
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)
frame_id = 0

print("🎥 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmentor.process(img_rgb)
    mask = result.segmentation_mask
    binary_mask = (mask > CONFIDENCE_THRESHOLD).astype(np.uint8) * 255

    # Check if any person is detected
    if np.count_nonzero(binary_mask) == 0:
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Apply mask and crop
    y_idx, x_idx = np.where(binary_mask == 255)
    x_min, x_max = max(x_idx.min() - MARGIN, 0), min(x_idx.max() + MARGIN, frame.shape[1])
    y_min, y_max = max(y_idx.min() - MARGIN, 0), min(y_idx.max() + MARGIN, frame.shape[0])
    person_only = cv2.bitwise_and(frame, frame, mask=binary_mask)
    cropped = person_only[y_min:y_max, x_min:x_max]

    # Show live output
    cv2.imshow("Live", cropped)

    # Save frame
    if SAVE_DETECTED_ONLY:
        output_path = Path(OUTPUT_FOLDER) / f"frame_{frame_id:04d}.jpg"
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        pil_img.save(output_path, format="JPEG", optimize=True, quality=95)
        print(f"💾 Saved: {output_path.name}")
        frame_id += 1

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Real-time session ended.")
