"""
RAFT optical-flow analysis of rigid tool motion on the CholecT50
challenge validation set, restricted to instrument bounding boxes.

Pipeline
--------
1. For every video, load the labels JSON and keep only frames that
   contain at least one instrument bounding box.
2. Compute RAFT optical flow (torchvision raft_large) for every pair of
   consecutive frames (both frames must exist; the pair is skipped if
   the frame numbering has a gap).
3. Compute per-pixel flow magnitude maps.
4. Mask the magnitude map with the union of the (slightly dilated)
   instrument bounding boxes of the first frame of the pair.
5. Accumulate magnitude histograms + running mean/std, per video and
   globally, and write everything to CSV / NPZ / PNG.

Requirements
------------
    pip install torch torchvision numpy opencv-python matplotlib tqdm
    (PyTorch with CUDA: https://pytorch.org/get-started/locally/)

Usage
-----
    python cholect50_raft_flow_analysis.py                # full run
    python cholect50_raft_flow_analysis.py --inspect      # only print one
                                                          # sample annotation
                                                          # to verify format
    python cholect50_raft_flow_analysis.py --videos VID74 VID75
