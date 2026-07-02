from torch.utils.data import Dataset
import h5py
import torch
import numpy as np

class RigidDataset(Dataset):

    def __init__(self, path, trajectories=None):

        self.file = h5py.File(path, "r")

        self.trajectories = []
        names = trajectories

        if names is None:
            names = list(self.file.keys())

        for name in names:
            group = self.file[name]
            self.trajectories.append({
                "name": name,
                "pose": group["pose"][:],
                "action": group["action"][:],
            })

        self.transitions = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["pose"])
            for frame in range(T-1):
                self.transitions.append(
                    (traj_idx, frame)
                )

        self.width=self.file.attrs["image_width"]
        self.height=self.file.attrs["image_height"]

        self.active_trajectory_indices = names


    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):

        traj_idx, frame_idx = self.transitions[idx]
        traj = self.trajectories[traj_idx]

        pose = traj["pose"][frame_idx]
        next_pose = traj["pose"][frame_idx+1]

        action = traj["action"][frame_idx]

        norm_pose = self.normalize_pose(pose)
        norm_next_pose = self.normalize_pose(next_pose)
       
        delta_pose = norm_next_pose - norm_pose

        T = len(traj["pose"])
        trajectory_t = frame_idx  / (T-1)

        return {
            "pose": torch.from_numpy(norm_pose).float(),
            "next_pose": torch.from_numpy(norm_next_pose).float(),
            "pose_raw": torch.from_numpy(pose).float(),
            "next_pose_raw": torch.from_numpy(next_pose).float(),

            "start_pose": self.normalize_pose(traj["pose"][0]),
            "goal_pose": self.normalize_pose(traj["pose"][-1]),

            "frame_idx": frame_idx,
            "trajectory_length": T,
            "trajectory_t": torch.tensor([trajectory_t], dtype=torch.float32),

            "delta_pose": torch.from_numpy(delta_pose).float(),
            "action": torch.from_numpy(action),

            "trajectory": traj["name"]
        }
    
    def get_trajectory(self, idx):
        traj = self.trajectories[idx]
        return {
            "name": traj["name"],
            "pose": self.normalize_pose(traj["pose"]),
            "pose_raw": traj["pose"],
            "action": traj["action"],
        }
    
    def set_active_primitives(self, trajectories=None):

        if trajectories is None:
            active = range(len(self.trajectories))
        elif isinstance(trajectories, str):
            active = [
                i for i, traj in enumerate(self.trajectories)
                if traj["name"] == trajectories
            ]
        else:
            active = [
                i for i, traj in enumerate(self.trajectories)
                if traj["name"] in trajectories
            ]

        self.transitions = []
        for traj_idx in active:
            T = len(self.trajectories[traj_idx]["pose"])
            for frame in range(T-1):
                self.transitions.append(
                    (traj_idx, frame)
                )

        self.active_trajectory_indices = active
    
    @property
    def num_trajectories(self):
        return len(self.trajectories)
    
    @property
    def trajectory_names(self):
        return [t["name"] for t in self.trajectories]
    
    def normalize_pose(self, pose, width=None, height=None):
        w = self.width if width is None else width
        h = self.height if height is None else height


        if isinstance(pose, torch.Tensor):
            pose = pose.clone()
        else:
            pose = pose.copy()

        pose[...,0] /= w
        pose[...,1] /= h
        pose[...,2] /= np.pi 

        return pose


    def denormalize_pose(self, pose, width=None, height=None):
        w = self.width if width is None else width
        h = self.height if height is None else height

        if isinstance(pose, torch.Tensor):
            pose = pose.clone()
        else:
            pose = pose.copy()

        pose[...,0] *= w
        pose[...,1] *= h
        pose[...,2] *= np.pi  

        return pose