# CholecSeg8k — SAM2 zero-shot tool tracking

**Approach:** no training. SAM2 (Meta, `sam2.1-hiera-tiny`) is a *promptable*
segmenter with memory-based **video propagation**. We seed frame 0 with the GT
mask of each instrument (reproducible), then let SAM2 propagate each mask across
all 80 consecutive frames of a clip.

> **Why a mask prompt, not a box:** a bounding box around a thin, diagonal
> instrument is mostly tissue, so SAM2 latches onto the tissue instead of the
> tool. A first attempt with box prompts collapsed to 0.15 tool IoU, with one
> object ballooning over the whole abdominal wall.

**Why this over the U-Net:** class-agnostic but temporally consistent — one
prompt tracks a tool through the whole clip, which is exactly what we need to
turn tool masks into smooth `[x, y, theta]` trajectories. It also sidesteps the
class-imbalance problem (the U-Net barely learned the rare `l_hook`).

**What it produces**
- `outputs/sam2_overlay.mp4` — raw | tracked-tool overlay over the clip.
- `outputs/sam2_trajectory.png` — extracted `[x, y, theta]` per tool over time.
- prints mean **tool IoU vs GT** (zero-shot), comparable to the U-Net's grasper IoU.

## Run
```bash
python experiments/segmentation/sam2_zeroshot/run.py
```
First run downloads the tiny checkpoint (~150 MB) from HuggingFace.

## Note
Seeding from the GT mask demonstrates *tracking* quality, not detection. On the
full unlabelled Cholec80 the seed would instead come from a click or the U-Net's
first-frame prediction.
