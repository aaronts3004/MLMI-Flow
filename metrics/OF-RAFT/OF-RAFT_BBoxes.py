import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


DATA_ROOT = Path("/tmp/datasets/cholec/CholecT50")
VIDEOS_DIR  = DATA_ROOT / "videos"    
LABELS_DIR  = DATA_ROOT / "labels"     
OUTPUT_DIR  = DATA_ROOT / "flow_analysis_output"


BBOX_LAYOUT       = "list"     
INSTRUMENT_ID_IDX = 1          
BBOX_IDXS         = (3, 4, 5, 6)   
DICT_INSTR_KEY    = "instrument"   
BBOX_NORMALIZED   = True       

DILATE_FRAC   = 0.05
                        
INFER_SIZE    = (480, 864)  
BATCH_SIZE    = 8      
HIST_BINS     = 200
HIST_RANGE    = (0.0, 100.0)  
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS      = (".png", ".jpg", ".jpeg")

#BBox handling

def load_annotations(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    for key in ("annotations", "labels", "frames"):
        if key in data and isinstance(data[key], dict):
            ann = data[key]
            break
    else:
        ann = data if isinstance(data, dict) else {}
    out = {}
    for k, v in ann.items():
        try:
            out[int(k)] = v
        except (ValueError, TypeError):
            continue
    return out


def parse_bboxes(instances, img_w: int, img_h: int):
    boxes = []
    if not isinstance(instances, list):
        return boxes
    for inst in instances:
        try:
            if BBOX_LAYOUT == "dict":
                vec = inst[DICT_INSTR_KEY]
                instr_id = vec[0]
                x, y, w, h = vec[-4:]
            else:
                instr_id = inst[INSTRUMENT_ID_IDX]
                x, y, w, h = (inst[i] for i in BBOX_IDXS)
        except (KeyError, IndexError, TypeError):
            continue
        if instr_id is None or float(instr_id) < 0:
            continue
        if any(v is None for v in (x, y, w, h)) or float(w) <= 0 or float(h) <= 0:
            continue
        x, y, w, h = float(x), float(y), float(w), float(h)
        if BBOX_NORMALIZED:
            x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h
        if DILATE_FRAC > 0:
            dx, dy = w * DILATE_FRAC, h * DILATE_FRAC
            x, y, w, h = x - dx, y - dy, w + 2 * dx, h + 2 * dy
        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(img_w, int(math.ceil(x + w)))
        y2 = min(img_h, int(math.ceil(y + h)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def boxes_to_mask(boxes, img_w: int, img_h: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
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

def process_video(model, vid: str, video_dir: Path, json_path: Path,
                  global_stats: FlowStats):
    ann = load_annotations(json_path)
    frames = list_frames(video_dir)
    if not frames or not ann:
        print(f"[{vid}] missing frames or labels -- skipped")
        return None

    probe = cv2.imread(str(next(iter(frames.values()))))
    H0, W0 = probe.shape[:2]

    tool_frames = {}
    for fid, inst in ann.items():
        boxes = parse_bboxes(inst, W0, H0)
        if boxes and fid in frames:
            tool_frames[fid] = boxes

    fids = sorted(frames)
    step = min(np.diff(fids)) if len(fids) > 1 else 1
    pairs = [(f, f + step) for f in sorted(tool_frames)
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
        batch1.append(load_image_tensor(frames[f1]))
        batch2.append(load_image_tensor(frames[f2]))
        masks.append(boxes_to_mask(tool_frames[f1], W0, H0))
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


def inspect_one_annotation():
    for json_path in sorted(LABELS_DIR.glob("*.json")):
        ann = load_annotations(json_path)
        for fid in sorted(ann):
            if isinstance(ann[fid], list) and ann[fid]:
                print(f"File : {json_path.name}")
                print(f"Frame: {fid}")
                print(f"Raw  : {json.dumps(ann[fid], indent=2)[:2000]}")
                return
    print("No non-empty annotations found -- check LABELS_DIR.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="print one sample annotation and exit")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="subset of video ids, e.g. VID74 VID75")
    args = ap.parse_args()

    if args.inspect:
        inspect_one_annotation()
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
        json_path = LABELS_DIR / f"{vid}.json"
        if not json_path.exists():
            cands = list(LABELS_DIR.glob(f"*{re.sub('[^0-9]', '', vid)}*.json"))
            if not cands:
                print(f"[{vid}] no labels JSON -- skipped")
                continue
            json_path = cands[0]

        stats = process_video(model, vid, vd, json_path, global_stats)
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
        sys.exit("No videos processed -- check paths / bbox config.")

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
