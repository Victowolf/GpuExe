import os
import cv2

INPUT_DIR = "dataset/LR_tiles"     # your HR folder (256×256)
OUTPUT_DIR = "dataset/LR_tiles2"    # your LR folder (created automatically)

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠ Could not read: {fname}")
        continue

    # Resize to 1/4 dimensions (256 -> 64)
    lr = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    # Save with SAME filename
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), lr)
    count += 1

print(f"✅ Downscaled {count} images from 256×256 → 64×64")
print(f"📁 Saved in: {OUTPUT_DIR}")
