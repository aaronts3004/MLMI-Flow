
import h5py

from src.datasets.rigid_dataset import RigidDataset
import matplotlib.pyplot as plt
import numpy as np

HDF5_PATH = "data/generated/rigid_dataset_isolated_primitives_100_trajectories.h5"
OBJECTIVE = "flow_matching"    # can be "autoregressive", "flow_matching"


##################################################################################
# First inspect the raw H5

print("\nInspecting the H5 raw dataset stored at: ", HDF5_PATH)
with h5py.File(HDF5_PATH, 'r') as file:

    print("\n All attributes in the dataset: ")
    for key, value in file.attrs.items():
        print(f"File Attribute - {key}: {value}")

    # Print the top-level keys (groups and datasets)
    print("\nAll top-level keys: ")
    print("Keys in the file:", list(file.keys()))
    

    print("\n Inspect a single dataset item (primitive=line)")
    line_group = file["line"]
    print("type(file['line'])= ", type(line_group))

    traj = line_group["trajectory_0000"]

    print("keys in line_group['trajectory_0000']: ", list(traj.keys()))

    print(f"traj['action'].shape={traj['action'].shape}")
    print(f"  traj['pose'].shape={traj['pose'].shape}")

    print("\n\n")

    for primitive_group_key in file.keys():
        primitive_group = file[primitive_group_key]
        poses = []
        for traj_name in primitive_group.keys():
            poses.append(primitive_group[traj_name]["pose"][:])
        poses = np.stack(poses)      # (N, T, 3)
        print(
            f"{primitive_group_key}: "
            f"{poses.std(axis=0).mean(axis=0)}"
        )
        plt.figure(figsize=(5,5))

        for traj_name in primitive_group.keys():
            pose = primitive_group[traj_name]["pose"][:]
            plt.plot(pose[:,0], pose[:,1], alpha=0.15)

        plt.gca().set_aspect("equal")
        plt.title(primitive_group_key)
        plt.show()

print("\n\n\n Inspect the torch.Dataset: ")
dataset = RigidDataset(HDF5_PATH, OBJECTIVE)


print("len(dataset)=", len(dataset))

dataset_sample = dataset[0]
print("\nGet a dataset sample as returned by __getitem__ (returns a dict)")
print("dataset_sample.keys()=", dataset_sample.keys())

print("dataset_sample['name']=", dataset_sample["name"])
print("dataset_sample['primitive']=", dataset_sample["primitive"])
print("dataset_sample['trajectory'].shape=", dataset_sample['trajectory'].shape)
print("dataset_sample['trajectory'][:10]=\n", dataset_sample['trajectory'][:10])

dataset_sample = dataset[100]
print("\nGet another dataset sample as returned by __getitem__ (returns a dict)")
print("dataset_sample.keys()=", dataset_sample.keys())

print("dataset_sample['name']=", dataset_sample["name"])
print("dataset_sample['primitive']=", dataset_sample["primitive"])
print("dataset_sample['trajectory'].shape=", dataset_sample['trajectory'].shape)