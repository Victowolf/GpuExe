import os
import cv2
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.network_swinir import SwinIR
from tqdm import tqdm

# ---------------------------------------------
# Dataset (paired HR/LR tiles) + 64px patch cropping
# ---------------------------------------------
class TileDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=64, scale=4):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale

        self.files = sorted([
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(".png") or f.lower().endswith(".jpg")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load HR/LR
        hr = cv2.imread(os.path.join(self.hr_dir, fname))
        lr = cv2.imread(os.path.join(self.lr_dir, fname))

        # Convert BGR→RGB
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        # Normalize
        hr = hr.astype(np.float32) / 255.0
        lr = lr.astype(np.float32) / 255.0

        # ---------------------------------------------
        # RANDOMLY CROP 64×64 LR patch + 256×256 HR patch
        # ---------------------------------------------
        H, W, _ = lr.shape
        ps = self.patch_size

        # pick random top-left
        x = random.randint(0, W - ps)
        y = random.randint(0, H - ps)

        # LR patch (64x64)
        lr_patch = lr[y:y+ps, x:x+ps]

        # HR patch corresponds to ×4 region in HR
        hr_x = x * self.scale
        hr_y = y * self.scale
        hr_patch = hr[hr_y:hr_y+ps*self.scale, hr_x:hr_x+ps*self.scale]

        # ---------------------------------------------
        # Convert CHW
        # ---------------------------------------------
        lr_patch = torch.from_numpy(lr_patch.transpose(2, 0, 1))
        hr_patch = torch.from_numpy(hr_patch.transpose(2, 0, 1))

        return lr_patch, hr_patch


# ---------------------------------------------
# Training function
# ---------------------------------------------
def train(config_path):

    # Load config
    with open(config_path, "r") as f:
        opt = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device}")

    # Dataset
    train_ds = TileDataset(
        opt["train_hr_dir"],
        opt["train_lr_dir"],
        patch_size=64,
        scale=4
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt["batch_size"],
        shuffle=True,
        num_workers=opt["num_workers"],
        pin_memory=True,
        drop_last=True
    )

    # ---------------------------------------------
    # 🚀 SwinIR-M REAL_SR ×4 — official architecture
    # ---------------------------------------------
    model = SwinIR(
        upscale=4,
        img_size=64,
        window_size=8,
        in_chans=3,
        img_range=1.0,
        depths=opt["depths"],
        embed_dim=opt["embed_dim"],
        num_heads=opt["num_heads"],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv"
    ).to(device)

    # Loss
    criterion = nn.L1Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt["lr"])

    # AMP
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(opt["save_model_dir"], exist_ok=True)

    # ---------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------
    for epoch in range(1, opt["epochs"] + 1):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{opt['epochs']}")

        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                sr = model(lr)
                loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        print(f"✔ Epoch {epoch} — Avg Loss: {epoch_loss / len(train_loader):.6f}")

        # Save model
        savepath = os.path.join(opt["save_model_dir"], f"epoch_{epoch}.pth")
        torch.save({"params_ema": model.state_dict()}, savepath)
        print(f"💾 Saved: {savepath}")


# ---------------------------------------------
# Entry
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    train(args.config)
