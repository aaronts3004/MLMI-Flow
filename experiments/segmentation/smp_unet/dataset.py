"""CholecSeg8k dataset for semantic segmentation.

Parses the `*_endo_watershed_mask.png` ground-truth masks into 13-class index
maps. The endoscope vignette (pixel 255) and stray 0 are mapped to ignore.
Splitting is done at the VIDEO level (consecutive frames are near-identical, so
a random frame split would leak).
"""

import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# repo root, so paths work regardless of the current working directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "kaggle", "cholecseg8k")

# watershed pixel value -> (class index, name)
WATERSHED_TO_CLASS = {
    50: (0, "background"),
    11: (1, "abdominal_wall"),
    21: (2, "liver"),
    13: (3, "gi_tract"),
    12: (4, "fat"),
    31: (5, "grasper"),
    23: (6, "connective_tissue"),
    24: (7, "blood"),
    25: (8, "cystic_duct"),
    32: (9, "l_hook"),
    22: (10, "gallbladder"),
    33: (11, "hepatic_vein"),
    5:  (12, "liver_ligament"),
}
NUM_CLASSES = 13
IGNORE_INDEX = 255
CLASS_NAMES = [name for _, name in sorted(
    ((idx, name) for idx, name in WATERSHED_TO_CLASS.values())
)]

# lookup table: watershed value -> class index (unknown values -> ignore)
_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _v, (_idx, _name) in WATERSHED_TO_CLASS.items():
    _LUT[_v] = _idx

# fixed video-level split (17 videos)
VAL_VIDEOS = ["video27", "video35"]
TEST_VIDEOS = ["video43", "video48", "video52"]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def all_videos(root=DATA_ROOT):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "video*")))


def split_videos(root=DATA_ROOT):
    vids = all_videos(root)
    val = [v for v in vids if v in VAL_VIDEOS]
    test = [v for v in vids if v in TEST_VIDEOS]
    train = [v for v in vids if v not in VAL_VIDEOS + TEST_VIDEOS]
    return train, val, test


def list_frames(videos, root=DATA_ROOT, stride=1):
    """Return (image_path, mask_path) pairs. `stride` subsamples frames within
    each clip (consecutive frames are highly redundant)."""
    items = []
    for v in videos:
        for clip in sorted(glob.glob(os.path.join(root, v, v + "_*"))):
            imgs = sorted(glob.glob(os.path.join(clip, "*_endo.png")))
            for img in imgs[::stride]:
                mask = img.replace("_endo.png", "_endo_watershed_mask.png")
                if os.path.exists(mask):
                    items.append((img, mask))
    return items


class CholecSeg8k(Dataset):
    def __init__(self, videos, root=DATA_ROOT, size=(384, 640), stride=1, train=False):
        self.items = list_frames(videos, root=root, stride=stride)
        self.size = size  # (H, W), both divisible by 32
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_p, mask_p = self.items[i]
        H, W = self.size

        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        ws = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        ws = cv2.resize(ws, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = _LUT[ws]

        if self.train and np.random.rand() < 0.5:      # horizontal flip
            img = img[:, ::-1]
            mask = mask[:, ::-1]

        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return img, mask
