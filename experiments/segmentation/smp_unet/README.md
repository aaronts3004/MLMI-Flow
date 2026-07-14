# CholecSeg8k — U-Net baseline (supervised)

**Approach:** supervised transfer learning. A U-Net with an ImageNet-pretrained
ResNet-34 encoder ([segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch))
is fine-tuned on the 13 pixel classes of CholecSeg8k. This validates *how good a
segmentation we can get* with a quick, standard model — the labelled stepping
stone before running it on the unlabelled full Cholec80 videos.

**Why this matters for the project:** the tool classes (`grasper`, `l_hook`)
give us instrument masks → `[x, y, theta]` tool trajectories (the real-data
analog of the synthetic primitives); the tissue classes are the first step
toward deformable structure.

## Data
Expects CholecSeg8k at `data/kaggle/cholecseg8k/`. Masks are the
`*_endo_watershed_mask.png` files; the 13 canonical pixel values are mapped to
class indices `0..12`, the endoscope vignette (255) is ignored.

## Key choices
- **Video-level split** (not frame-level) — consecutive frames are near-identical
  and would leak. `val = {27, 35}`, `test = {43, 48, 52}`, rest = train.
- **Frame stride 5** — subsamples redundant consecutive frames for speed.
- **Metric:** per-class IoU + mIoU on held-out videos (ignore = 255).

## Run
```bash
# from the repo root, with the venv active
python experiments/segmentation/smp_unet/train.py
```
Outputs: `outputs/best_unet.pt` and `outputs/preview_*.png` (image | GT | pred).

## Known limitation / next knob
Severe class imbalance — `grasper`/`l_hook` are ~0.6% / 0.3% of pixels, so plain
cross-entropy under-segments exactly the tool classes we care about. First
improvement to try: class-weighted or Dice/Focal loss.
