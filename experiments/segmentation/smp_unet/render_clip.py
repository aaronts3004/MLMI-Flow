"""Render a segmentation overlay video for one clip using the trained U-Net.

Runs inference over all 80 consecutive frames of a clip (from a held-out TEST
video by default) and writes an mp4 where the prediction is alpha-blended over
the RGB frame -- shows both quality and temporal stability.
"""

import glob
import os
import re

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from dataset import DATA_ROOT, TEST_VIDEOS, NUM_CLASSES, _MEAN, _STD
from train import get_device, SIZE, OUT_DIR, PALETTE


def pick_clip():
    """Test-video clip with the most tool (grasper/l_hook) pixels in its GT."""
    best = (-1, None)
    for v in TEST_VIDEOS:
        for clip in sorted(glob.glob(os.path.join(DATA_ROOT, v, v + "_*"))):
            masks = sorted(glob.glob(os.path.join(clip, "*_endo_watershed_mask.png")))
            if not masks:
                continue
            ws = cv2.imread(masks[len(masks) // 2], cv2.IMREAD_GRAYSCALE)
            tool = np.mean((ws == 31) | (ws == 32))
            if tool > best[0]:
                best = (tool, clip)
    return best[1]


def main():
    device = get_device()
    model = smp.Unet("resnet34", encoder_weights=None, classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_unet.pt"), map_location=device))
    model.eval()

    clip = pick_clip()
    print("clip:", clip)
    frames = sorted(
        glob.glob(os.path.join(clip, "*_endo.png")),
        key=lambda p: int(re.search(r"frame_(\d+)_", p).group(1)),
    )

    H, W = SIZE
    out_path = os.path.join(OUT_DIR, "segmentation_overlay.mp4")
    writer = None

    with torch.no_grad():
        for fp in frames:
            bgr = cv2.imread(fp)
            oh, ow = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            x = (x - _MEAN) / _STD
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

            pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
            color = PALETTE[pred][..., ::-1]                      # to BGR

            overlay = cv2.addWeighted(bgr, 0.55, color, 0.45, 0)
            combo = np.hstack([bgr, overlay])                    # raw | overlay

            if writer is None:
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (combo.shape[1], combo.shape[0])
                )
            writer.write(combo)

    writer.release()
    print("Saved", out_path, "|", len(frames), "frames")


if __name__ == "__main__":
    main()
