import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


DATA_ROOT = Path(<define path>)
VIDEOS_DIR  = DATA_ROOT / "videos"
OUTPUT_DIR  = DATA_ROOT / "flow_analysis_output"


MASKS_DIR   = DATA_ROOT / "masks"
MASK_EXTS   = (".png", ".jpg", ".jpeg", ".npy", ".npz")
MASK_NPZ_KEY = "mask"   # key to read if a mask file is a .npz archive

MASK_DILATE_PX = 0   # optional morphological dilation (pixels) of the
                     # SAM2 mask before pooling flow, e.g. to catch
                     # motion blur right at the tool boundary. 0 = off,
                     # since SAM2 masks are already tight vs. bboxes.

INFER_SIZE    = (480, 864)  
BATCH_SIZE    = 8      
HIST_BINS     = 200
HIST_RANGE    = (0.0, 100.0)  
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS      = (".png", ".jpg", ".jpeg")

#SAM2 mask handling

def list_masks(masks_dir: Path) -> dict:
    """Return {frame_id(int): path}, mirroring list_frames()."""
    masks = {}
    if not masks_dir.is_dir():
        return masks
    for p in masks_dir.iterdir():
        if p.suffix.lower() in MASK_EXTS:
            m = re.search(r"(\d+)", p.stem)
            if m:
                masks[int(m.group(1))] = p
    return masks


def load_mask(path: Path) -> np.ndarray:
    """Load a precomputed SAM2 mask as a boolean (H, W) array.

    Supports single-channel/RGB image masks (nonzero = tool) as well as
    .npy / .npz arrays, so it works regardless of which format the SAM2
    export used.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as npz:
            key = MASK_NPZ_KEY if MASK_NPZ_KEY in npz else list(npz.keys())[0]
            arr = npz[key]
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"Could not read mask {path}")
        if arr.ndim == 3:
            arr = arr[..., 0]
    return arr > 0


def prepare_mask(mask: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Resize to the frame resolution (in case masks were saved at a
    different resolution) and apply optional dilation."""
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    if MASK_DILATE_PX > 0:
        k = MASK_DILATE_PX * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask

#Frames

def list_frames(video_dir: Path) -> dict:
    """Return {frame_id(int): path}, frame id parsed from the filename digits."""
    frames = {}
    for p in video_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            m = re.search(r"(\d+)", p.stem)
            if m:
                frames[int(m.group(1))] = p
    return frames


def load_image_tensor(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t * 2.0 - 1.0  # RAFT expects [-1, 1]

#Statistics

class FlowStats:
    def __init__(self):
        self.hist = np.zeros(HIST_BINS, dtype=np.int64)
        self.n = 0
        self.s = 0.0   
        self.s2 = 0.0   
        self.mx = 0.0
        self.n_pairs = 0

    def update(self, mags: np.ndarray):
        if mags.size == 0:
            return
        clipped = np.clip(mags, HIST_RANGE[0], HIST_RANGE[1] - 1e-6)
        h, _ = np.histogram(clipped, bins=HIST_BINS, range=HIST_RANGE)
        self.hist += h
        self.n += mags.size
        self.s += float(mags.sum(dtype=np.float64))
        self.s2 += float(np.square(mags, dtype=np.float64).sum())
        self.mx = max(self.mx, float(mags.max()))

    @property
    def mean(self):
        return self.s / self.n if self.n else float("nan")

    @property
    def std(self):
        if not self.n:
            return float("nan")
        var = max(self.s2 / self.n - self.mean ** 2, 0.0)
        return math.sqrt(var)

    def percentile(self, q):
        """Approximate percentile from the histogram."""
        if not self.n:
            return float("nan")
        edges = np.linspace(*HIST_RANGE, HIST_BINS + 1)
        cdf = np.cumsum(self.hist) / self.n
        idx = int(np.searchsorted(cdf, q / 100.0))
        return float(edges[min(idx + 1, HIST_BINS)])

# RAFT

@torch.no_grad()
def run_raft_batch(model, im1: torch.Tensor, im2: torch.Tensor,
                   orig_hw) -> torch.Tensor:
    H0, W0 = orig_hw
    Hi, Wi = INFER_SIZE
    im1r = F.interpolate(im1, size=(Hi, Wi), mode="bilinear", align_corners=False)
    im2r = F.interpolate(im2, size=(Hi, Wi), mode="bilinear", align_corners=False)
    flow = model(im1r.to(DEVICE), im2r.to(DEVICE))[-1]         
    flow = F.interpolate(flow, size=(H0, W0), mode="bilinear",
                         align_corners=False)
    flow[:, 0] *= W0 / Wi   
    flow[:, 1] *= H0 / Hi
    return torch.linalg.vector_norm(flow, dim=1).cpu()

# single video processing

def process_video(model, vid: str, video_dir: Path, masks_dir: Path,
                  global_stats: FlowStats):
    frames = list_frames(video_dir)
    mask_paths = list_masks(masks_dir)
    if not frames or not mask_paths:
        print(f"[{vid}] missing frames or SAM2 masks -- skipped")
        return None

    probe = cv2.imread(str(next(iter(frames.values()))))
    H0, W0 = probe.shape[:2]

    fids = sorted(frames)
    step = min(np.diff(fids)) if len(fids) > 1 else 1
    # a mask file existing for f1 stands in for "tool annotated here";
    # frames with an empty mask (no tool pixels) are skipped below once
    # the mask is actually loaded, same as before with empty bbox lists.
    pairs = [(f, f + step) for f in sorted(mask_paths)
             if (f + step) in frames]
    if not pairs:
        print(f"[{vid}] no usable frame pairs -- skipped")
        return None

    stats = FlowStats()
    batch1, batch2, masks = [], [], []

    def flush():
        if not batch1:
            return
        t1 = torch.stack(batch1)
        t2 = torch.stack(batch2)
        mags = run_raft_batch(model, t1, t2, (H0, W0)).numpy()
        for m, msk in zip(mags, masks):
            vals = m[msk]
            stats.update(vals)
            global_stats.update(vals)
        stats.n_pairs += len(batch1)
        global_stats.n_pairs += len(batch1)
        batch1.clear(); batch2.clear(); masks.clear()

    from tqdm import tqdm
    for f1, f2 in tqdm(pairs, desc=vid, unit="pair"):
        mask = prepare_mask(load_mask(mask_paths[f1]), W0, H0)
        if not mask.any():
            continue
        batch1.append(load_image_tensor(frames[f1]))
        batch2.append(load_image_tensor(frames[f2]))
        masks.append(mask)
        if len(batch1) >= BATCH_SIZE:
            flush()
    flush()
    return stats


def save_histogram_plot(stats: FlowStats, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.linspace(*HIST_RANGE, HIST_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    density = stats.hist / max(stats.n, 1) / (edges[1] - edges[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, density, width=edges[1] - edges[0], color="#4477aa")
    ax.set_xlabel("Flow magnitude (px / frame step)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.axvline(stats.mean, color="crimson", ls="--",
               label=f"mean = {stats.mean:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def inspect_one_mask():
    for masks_dir in sorted(d for d in MASKS_DIR.iterdir() if d.is_dir()):
        for fid, path in sorted(list_masks(masks_dir).items()):
            mask = load_mask(path)
            if mask.any():
                print(f"Video: {masks_dir.name}")
                print(f"Frame: {fid}")
                print(f"Path : {path}")
                print(f"Shape: {mask.shape}  dtype: {mask.dtype}  "
                      f"tool_px: {int(mask.sum())} / {mask.size}")
                return
    print("No non-empty masks found -- check MASKS_DIR.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="print one sample SAM2 mask and exit")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="subset of video ids, e.g. VID74 VID75")
    args = ap.parse_args()

    if args.inspect:
        inspect_one_mask()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEVICE).eval()

    video_dirs = sorted(d for d in VIDEOS_DIR.iterdir() if d.is_dir())
    if args.videos:
        video_dirs = [d for d in video_dirs if d.name in set(args.videos)]

    global_stats = FlowStats()
    rows = []
    for vd in video_dirs:
        vid = vd.name
        masks_dir = MASKS_DIR / vid
        if not masks_dir.is_dir():
            # fall back to a fuzzy match in case the SAM2 export used a
            # slightly different directory name for this video id
            cands = [d for d in MASKS_DIR.iterdir()
                     if d.is_dir() and re.sub('[^0-9]', '', vid) in d.name]
            if not cands:
                print(f"[{vid}] no SAM2 masks dir -- skipped")
                continue
            masks_dir = cands[0]

        stats = process_video(model, vid, vd, masks_dir, global_stats)
        if stats is None:
            continue
        rows.append({
            "video": vid, "n_pairs": stats.n_pairs, "n_pixels": stats.n,
            "mean": round(stats.mean, 4), "std": round(stats.std, 4),
            "median": round(stats.percentile(50), 4),
            "p90": round(stats.percentile(90), 4),
            "p99": round(stats.percentile(99), 4),
            "max": round(stats.mx, 4),
        })
        np.savez(OUTPUT_DIR / f"{vid}_hist.npz", hist=stats.hist,
                 bin_edges=np.linspace(*HIST_RANGE, HIST_BINS + 1))
        save_histogram_plot(stats, f"{vid} -- tool-region flow magnitude",
                            OUTPUT_DIR / f"{vid}_hist.png")
        print(f"[{vid}] pairs={stats.n_pairs}  mean={stats.mean:.3f}  "
              f"std={stats.std:.3f}  median={stats.percentile(50):.3f}")

    if not rows:
        sys.exit("No videos processed -- check VIDEOS_DIR / MASKS_DIR paths.")

    rows.append({
        "video": "GLOBAL", "n_pairs": global_stats.n_pairs,
        "n_pixels": global_stats.n,
        "mean": round(global_stats.mean, 4),
        "std": round(global_stats.std, 4),
        "median": round(global_stats.percentile(50), 4),
        "p90": round(global_stats.percentile(90), 4),
        "p99": round(global_stats.percentile(99), 4),
        "max": round(global_stats.mx, 4),
    })
    with open(OUTPUT_DIR / "flow_statistics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    np.savez(OUTPUT_DIR / "GLOBAL_hist.npz", hist=global_stats.hist,
             bin_edges=np.linspace(*HIST_RANGE, HIST_BINS + 1))
    save_histogram_plot(global_stats,
                        "All validation videos -- tool-region flow magnitude",
                        OUTPUT_DIR / "GLOBAL_hist.png")

    print("\n================ SUMMARY ================")
    for r in rows:
        print(f"{r['video']:>8}: mean={r['mean']:.3f}  std={r['std']:.3f}  "
              f"median={r['median']:.3f}  p90={r['p90']:.3f}  "
              f"pairs={r['n_pairs']}")
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
