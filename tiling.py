import os
import cv2
import numpy as np

HR_DIR = "dataset/HR"
LR_DIR = "dataset/LR"

OUT_HR = "dataset/HR_tiles"
OUT_LR = "dataset/LR_tiles"

os.makedirs(OUT_HR, exist_ok=True)
os.makedirs(OUT_LR, exist_ok=True)

TILE = 256  # tile size
counter = 1  # global tile counter


def tile_image_pair(hr_img, lr_img, tile, prefix_num):
    h, w = hr_img.shape[:2]

    tiles = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            hr_crop = hr_img[y:y+tile, x:x+tile]
            lr_crop = lr_img[y:y+tile, x:x+tile]
            tiles.append((hr_crop, lr_crop))
    return tiles


print("🔹 Splitting HR/LR image pairs into tiles...")

for fname in os.listdir(HR_DIR):
    if not fname.lower().endswith(("jpg", "png", "jpeg")):
        continue

    base = os.path.splitext(fname)[0]

    hr = cv2.imread(os.path.join(HR_DIR, fname))
    lr = cv2.imread(os.path.join(LR_DIR, fname))

    if hr is None or lr is None:
        print(f"⚠ Skipping {fname}, failed to read.")
        continue

    # Tile this HR/LR pair
    tiles = tile_image_pair(hr, lr, TILE, counter)

    # Save tiles sequentially
    for hr_tile, lr_tile in tiles:
        hr_path = os.path.join(OUT_HR, f"{counter}.png")
        lr_path = os.path.join(OUT_LR, f"{counter}.png")

        cv2.imwrite(hr_path, hr_tile)
        cv2.imwrite(lr_path, lr_tile)

        counter += 1

print(f"✔ Done! Saved {counter-1} tiles.")
