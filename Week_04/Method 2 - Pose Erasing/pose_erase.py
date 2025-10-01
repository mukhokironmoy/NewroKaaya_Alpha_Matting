import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from PIL import Image

# === CONFIG ===
OUTPUT_FOLDER = r"Method 2 - Pose Erasing\pose_erased_out"
MARGIN_CM = 1
DPI = 96  # Dots per inch, used for cm-to-px conversion
CM_TO_PX = int(MARGIN_CM * DPI / 2.54)

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Create output folder
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
frame_id = 0

print("🎥 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    # Create a blank black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if result.pose_landmarks:
        points = []
        for lm in result.pose_landmarks.landmark:
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                # Draw a filled circle around each landmark
                cv2.circle(mask, (cx, cy), CM_TO_PX, 255, -1)

        # Optional: Connect joints with lines (to form shape)
        joint_pairs = mp_pose.POSE_CONNECTIONS
        for p1, p2 in joint_pairs:
            pt1 = result.pose_landmarks.landmark[p1]
            pt2 = result.pose_landmarks.landmark[p2]
            if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(mask, (x1, y1), (x2, y2), 255, CM_TO_PX)

        # Smooth + dilate the mask
        mask = cv2.dilate(mask, np.ones((CM_TO_PX, CM_TO_PX), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Convert mask to 0–1 alpha channel
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0, 1)

        # Apply alpha to each channel
        b, g, r = cv2.split(frame)
        rgba = cv2.merge((b, g, r, (alpha * 255).astype(np.uint8)))

        # Save as PNG
        output_path = Path(OUTPUT_FOLDER) / f"frame_{frame_id:04d}.png"
        pil_img = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
        pil_img.save(output_path)
        print(f"💾 Saved: {output_path.name}")
        frame_id += 1

        # Show preview
        print("Alpha max:", np.max(alpha), "| Alpha min:", np.min(alpha))
        print("RGBA shape:", rgba.shape)

        # cv2.imshow("Pose Alpha Matte", rgba)
        # Convert RGBA (from PIL) → BGRA (for OpenCV)
        # preview_bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        # print("showing preview")
        # cv2.imshow("Pose Alpha Matte", preview_bgra)

        # Convert alpha to 1-channel mask (float32 in range 0–1)
        alpha = np.clip(alpha, 0, 1)
        alpha_3c = np.stack([alpha]*3, axis=-1)  # Shape (H, W, 3)

        # Extract foreground
        foreground = frame.astype(np.float32) * alpha_3c

        # Create solid white background
        white_bg = np.ones_like(frame, dtype=np.float32) * 255

        # Combine using per-pixel alpha
        preview = foreground + white_bg * (1 - alpha_3c)

        # Convert to uint8 for display
        preview = preview.astype(np.uint8)
        cv2.imshow("Pose Alpha Matte", preview)

        
    else:
        print("showing frame not processed")
        cv2.imshow("Pose Alpha Matte", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
