"""SAM2 zero-shot tool tracking on one CholecSeg8k clip.

Seeds frame-0 with the GT mask of each instrument, propagates the masks across
the 80 frames, renders an overlay video, extracts an [x,y,theta] trajectory per
tool, and reports mean tool IoU vs GT.
"""

import glob
import os
import re
import tempfile

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from sam2.sam2_video_predictor import SAM2VideoPredictor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "kaggle", "cholecseg8k")
TEST_VIDEOS = ["video43", "video48", "video52"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
TOOL_VALS = (31, 32)                      # grasper, l_hook
OBJ_COLORS = {1: (60, 220, 60), 2: (60, 120, 240)}   # BGR


def pick_clip():
    best = (-1, None)
    for v in TEST_VIDEOS:
        for clip in sorted(glob.glob(os.path.join(DATA_ROOT, v, v + "_*"))):
            masks = sorted(glob.glob(os.path.join(clip, "*_endo_watershed_mask.png")))
            if not masks:
                continue
            ws = cv2.imread(masks[len(masks) // 2], cv2.IMREAD_GRAYSCALE)
            tool = np.mean(np.isin(ws, TOOL_VALS))
            if tool > best[0]:
                best = (tool, clip)
    return best[1]


def frame_paths(clip):
    return sorted(glob.glob(os.path.join(clip, "*_endo.png")),
                  key=lambda p: int(re.search(r"frame_(\d+)_", p).group(1)))


def orientation(mask):
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    pts = np.stack([xs - cx, ys - cy], 1).astype(np.float32)
    _, v = np.linalg.eigh(pts.T @ pts)
    major = v[:, -1]
    return cx, cy, np.arctan2(major[1], major[0])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny", device=device)
    print("device:", device)

    clip = pick_clip()
    print("clip:", clip)
    frames = frame_paths(clip)
    masks_gt = [p.replace("_endo.png", "_endo_watershed_mask.png") for p in frames]

    # seed frame: first frame that contains a tool
    seed_idx = next(i for i, m in enumerate(masks_gt)
                    if np.isin(cv2.imread(m, 0), TOOL_VALS).any())
    seed_ws = cv2.imread(masks_gt[seed_idx], 0)
    # Seed with the GT mask, not a box: a box around a thin diagonal instrument is
    # mostly tissue, so SAM2 latches onto the tissue instead of the tool.
    seed_masks = {}                       # obj_id -> boolean GT mask
    for oid, val in zip((1, 2), TOOL_VALS):
        m = seed_ws == val
        if m.sum() > 100:
            seed_masks[oid] = m
    print(f"seed frame {seed_idx}, objects: {list(seed_masks)}")

    # write frames as jpg for SAM2's video loader
    tmp = tempfile.mkdtemp()
    for i, fp in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), cv2.imread(fp))

    state = predictor.init_state(video_path=tmp)
    for oid, gt_mask in seed_masks.items():
        predictor.add_new_mask(state, frame_idx=seed_idx, obj_id=oid, mask=gt_mask)

    segs = {}
    for f_idx, obj_ids, logits in predictor.propagate_in_video(state):
        segs[f_idx] = {oid: (logits[i, 0] > 0).cpu().numpy() for i, oid in enumerate(obj_ids)}
    # also propagate backwards to cover frames before the seed
    for f_idx, obj_ids, logits in predictor.propagate_in_video(state, reverse=True):
        segs.setdefault(f_idx, {oid: (logits[i, 0] > 0).cpu().numpy() for i, oid in enumerate(obj_ids)})

    os.makedirs(OUT_DIR, exist_ok=True)
    H, W = cv2.imread(frames[0]).shape[:2]
    writer = cv2.VideoWriter(os.path.join(OUT_DIR, "sam2_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (2 * W, H))
    traj = {oid: [] for oid in seed_masks}
    ious = []
    for i, fp in enumerate(frames):
        bgr = cv2.imread(fp)
        overlay = bgr.copy()
        pred_tool = np.zeros((H, W), bool)
        for oid in seed_masks:
            m = segs.get(i, {}).get(oid)
            if m is None or m.sum() < 30:
                traj[oid].append((np.nan, np.nan, np.nan))
                continue
            pred_tool |= m
            overlay[m] = (0.45 * np.array(OBJ_COLORS[oid]) + 0.55 * overlay[m]).astype(np.uint8)
            traj[oid].append(orientation(m))
        gt_tool = np.isin(cv2.imread(masks_gt[i], 0), TOOL_VALS)
        inter = (pred_tool & gt_tool).sum()
        union = (pred_tool | gt_tool).sum()
        if union > 0:
            ious.append(inter / union)
        writer.write(np.hstack([bgr, overlay]))
    writer.release()
    print(f"mean tool IoU vs GT (zero-shot): {np.mean(ious):.3f}  over {len(ious)} frames")

    # trajectory plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for oid in seed_masks:
        a = np.array(traj[oid]); t = np.arange(len(a))
        ax[0].plot(a[:, 0], a[:, 1], "-o", ms=3, label=f"obj{oid}")
        ax[1].plot(t, a[:, 0], label=f"obj{oid} x"); ax[1].plot(t, a[:, 1], label=f"obj{oid} y")
        ax[2].plot(t, np.degrees(a[:, 2]), ".-", label=f"obj{oid}")
    ax[0].set_xlim(0, W); ax[0].set_ylim(H, 0); ax[0].set_title("tool centroid path"); ax[0].legend()
    ax[1].set_title("x,y over time"); ax[1].legend(); ax[2].set_title("theta (deg)"); ax[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "sam2_trajectory.png"), dpi=80)
    print("Saved outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
