import h5py
import numpy as np
from generate_primitives import PrimitivesGenerator

import h5py
from pathlib import Path

from generate_primitives import PrimitivesGenerator

# --------------------------------------------

H = 256
W = 256
T = 100                 # num time-steps per trajectory

NUM_TRAJECTORIES = 100  # total num trajectories


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)
OUTFILE = GENERATED_DIR / f"rigid_dataset_isolated_primitives_{NUM_TRAJECTORIES}_trajectories.h5"

# --------------------------------------------

generator = PrimitivesGenerator(H,W)
primitives = [
    ("line", generator.generate_line),
    ("circle", generator.generate_circle),
    ("arc", generator.generate_arc),
    ("rectangle", generator.generate_rectangle),
    ("zigzag", generator.generate_zigzag),
    ("figure8", generator.generate_figure8),
]

# --------------------------------------------

with h5py.File(OUTFILE,"w") as f:

    f.attrs["image_width"] = W
    f.attrs["image_height"] = H
    f.attrs["num_trajectories"] = NUM_TRAJECTORIES
    f.attrs["steps_per_traj"] = T

    for primitive_name, primitive_fn in primitives:

        pose = primitive_fn(T)                    # ground-truth pose trajectory
        action = generator.compute_action(pose)   # delta_pose for all timesteps

        group = f.create_group(primitive_name)

        group.create_dataset("pose", data=pose)
        group.create_dataset("action", data=action)

        group.attrs["primitive"] = primitive_name
        group.attrs["length"] = T

print("Dataset written.")