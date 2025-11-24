import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from models.network_swinir import SwinIR

# -------------------------------
#  Dataset loader for HR/LR tiles
# -------------------------------
class TileDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir

        self.files = sorted([f for f in os.listdir(hr_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        hr = cv2.imread(os.path.join(self.hr_dir, fname))
        lr = cv2.imread(os.path.join(self.lr_dir, fname))

        hr = hr.astype(np.float32) / 255.0
        lr = lr.astype(np.float32) / 255.0

        hr = torch.from_numpy(hr.transpose(2, 0, 1))
        lr = torch.from_numpy(lr.transpose(2, 0, 1))

        return lr, hr


# -------------------------------
#  Training Loop
# -------------------------------
def train(config_path):

    # Load JSON config
    with open(config_path, "r") as f:
        opt = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device}")

    # Dataset
    train_ds = TileDataset(opt["train_hr_dir"], opt["train_lr_dir"])
    train_loader = DataLoader(
        train_ds,
        batch_size=opt["batch_size"],
        shuffle=True,
        num_workers=opt["num_workers"],
        pin_memory=True
    )

    # SwinIR model (light version for 6GB GPU)
    model = SwinIR(
        upscale=opt["scale"],
        img_size=opt["tile_size"],
        window_size=8,
        in_chans=3,
        embed_dim=opt["embed_dim"],
        depths=opt["depths"],
        num_heads=opt["num_heads"],
    ).to(device)

    # Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt["lr"])

    # AMP for lower VRAM usage
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(opt["save_model_dir"], exist_ok=True)

    # ---------------------------
    #    TRAINING
    # ---------------------------
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

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        print(f"✔ Epoch {epoch} finished — Avg Loss: {epoch_loss / len(train_loader):.6f}")

        # Save model every epoch
        savepath = os.path.join(opt["save_model_dir"], f"epoch_{epoch}.pth")
        torch.save({"params": model.state_dict()}, savepath)
        print(f"💾 Saved: {savepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    train(args.config)
