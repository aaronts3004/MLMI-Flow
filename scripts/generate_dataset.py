import h5py
import numpy as np
from src.utils.paths import GENERATED
from scripts.generate_primitives import PrimitivesGenerator
import os

# --------------------------------------------

H = 256
W = 256
T = 100                 # num time-steps per trajectory

NUM_TRAJECTORIES = 100  # total num trajectories
OUTFILE = GENERATED / f"rigid_dataset_isolated_primitives_{NUM_TRAJECTORIES}_trajectories.h5"

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

    print(f"Generating {NUM_TRAJECTORIES} per primitive with {T} timesteps per trajectory")

    for primitive_name, primitive_fn in primitives:

        primitive_group = f.create_group(primitive_name)

        for traj_idx in range(NUM_TRAJECTORIES):

            pose = primitive_fn(T)
            action = generator.compute_action(pose)

            traj_group = primitive_group.create_group(
                f"trajectory_{traj_idx:04d}"
            )

            traj_group.create_dataset("pose", data=pose)
            traj_group.create_dataset("action", data=action)

            traj_group.attrs["length"] = T

print("Dataset written to OUTFILE=", OUTFILE)
