"""Train a U-Net (smp, ResNet-34 encoder) on CholecSeg8k and report mIoU.

Approach: supervised transfer learning. Encoder is ImageNet-pretrained; we
fine-tune on CholecSeg8k's 13 pixel classes, validate per-class IoU + mIoU on
held-out videos, and dump a few prediction overlays to eyeball quality.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import (
    CholecSeg8k, split_videos, NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES, REPO_ROOT,
)

# ---------------- config ----------------
EPOCHS = 8
BATCH_SIZE = 8
SIZE = (256, 448)          # (H, W), divisible by 32; raise for better mIoU
STRIDE = 10                # use every 10th frame (consecutive frames are redundant)
LR = 3e-4
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
# ----------------------------------------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fast_hist(pred, gt, n):
    valid = (gt >= 0) & (gt < n)           # excludes ignore (255)
    return np.bincount(n * gt[valid] + pred[valid], minlength=n * n).reshape(n, n)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, mask in loader:
        logits = model(img.to(device))
        pred = logits.argmax(1).cpu().numpy()
        hist += fast_hist(pred.ravel(), mask.numpy().ravel(), NUM_CLASSES)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-9)
    return iou


PALETTE = np.array([
    [0, 0, 0], [180, 120, 60], [120, 20, 20], [200, 150, 90], [220, 200, 120],
    [60, 220, 60], [230, 160, 180], [220, 30, 30], [180, 100, 220],
    [30, 220, 120], [230, 210, 100], [90, 90, 230], [150, 230, 230],
], dtype=np.uint8)


@torch.no_grad()
def save_previews(model, dataset, device, n=6):
    os.makedirs(OUT_DIR, exist_ok=True)
    model.eval()
    idxs = np.linspace(0, len(dataset) - 1, n).astype(int)
    for k, i in enumerate(idxs):
        img, mask = dataset[i]
        pred = model(img.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        rgb = ((img.numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] +
                [0.485, 0.456, 0.406]) * 255).clip(0, 255).astype(np.uint8)
        gt_col = PALETTE[np.clip(mask.numpy(), 0, NUM_CLASSES - 1)]
        pr_col = PALETTE[pred]
        combo = np.hstack([rgb[..., ::-1], gt_col[..., ::-1], pr_col[..., ::-1]])
        cv2.imwrite(os.path.join(OUT_DIR, f"preview_{k}.png"), combo)
    print(f"Saved previews to {OUT_DIR} (columns: image | GT | pred)")


def main():
    device = get_device()
    print("Device:", device)

    train_v, val_v, test_v = split_videos()
    print(f"videos  train={train_v}\n        val={val_v}\n        test={test_v}")

    train_ds = CholecSeg8k(train_v, size=SIZE, stride=STRIDE, train=True)
    val_ds = CholecSeg8k(val_v, size=SIZE, stride=STRIDE)
    print(f"frames  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_miou = 0.0
    os.makedirs(OUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            loss = criterion(model(img), mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        iou = evaluate(model, val_loader, device)
        miou = float(np.nanmean(iou))
        print(f"\nEpoch {epoch:02d} | train loss {running / len(train_loader):.4f} | mIoU {miou:.4f}")
        for name, c in sorted(zip(CLASS_NAMES, iou), key=lambda x: -x[1]):
            print(f"    {name:20s} IoU {c:.3f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_unet.pt"))

    print(f"\nBest val mIoU: {best_miou:.4f}")
    save_previews(model, val_ds, device)


if __name__ == "__main__":
    main()
