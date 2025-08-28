from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
import os

# === CONFIG ===
INPUT_FOLDER = Path(r"Test_input")
OUTPUT_FOLDER = r"Method 4 - Batch segmentation and compress\segment_and_compress_out"
CONFIDENCE_THRESHOLD = 0.1
MARGIN = 30
CAPTURE_DURATION = 10  # seconds
FPS = 30

#Helper functions
def clean_dir(folder_path):
    folder_path = Path(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# === INIT DIRECTORIES ===
Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# === 1. WEBCAM CAPTURE ===
def use_webcam():
    print("🎥 Starting webcam for 10 seconds...")
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to grab frame.")
            break

        timestamp = time.time()
        if timestamp - start_time > CAPTURE_DURATION:
            break

        filename = f"frame_{frame_count:03d}.jpg"
        cv2.imwrite(str(Path(INPUT_FOLDER) / filename), frame)
        frame_count += 1

        # Optional: Display live preview
        cv2.imshow("Recording...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved {frame_count} frames to {INPUT_FOLDER}")




# === 2. MEDIAPIPE SEGMENTATION ===
def using_segmentation():
    print("🧠 Starting segmentation and background removal...")

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    clean_dir(OUTPUT_FOLDER)

    for img_path in Path(INPUT_FOLDER).glob("*.[jp][pn]g"):
        print(f"Processing: {img_path.name}")
        
        # Read image using OpenCV
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Segment
        result = segmentor.process(img_rgb)
        mask = result.segmentation_mask

        # Create binary mask
        binary_mask = (mask > CONFIDENCE_THRESHOLD).astype(np.uint8) * 255

        if np.count_nonzero(binary_mask) == 0:
            print("No person detected. Skipping.")
            continue

        # Get bounding box with margin
        y_indices, x_indices = np.where(binary_mask == 255)
        x_min, x_max = max(x_indices.min() - MARGIN, 0), min(x_indices.max() + MARGIN, img.shape[1])
        y_min, y_max = max(y_indices.min() - MARGIN, 0), min(y_indices.max() + MARGIN, img.shape[0])

        # Apply mask and crop
        person_only = cv2.bitwise_and(img, img, mask=binary_mask)
        cropped = person_only[y_min:y_max, x_min:x_max]

        # Convert to PIL and save
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        output_path = Path(OUTPUT_FOLDER) / img_path.name
        pil_img.save(output_path, format="JPEG", optimize=True, quality=95)

    print("🎉 All frames processed and saved to:", OUTPUT_FOLDER)



#Main call
if __name__ == '__main__':
    using_segmentation()
