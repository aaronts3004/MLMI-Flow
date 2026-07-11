from torch.utils.data import Dataset
import h5py
import torch
import numpy as np

class RigidDataset(Dataset):

    def __init__(self, path, train_objective, trajectories=None,):

        self.file = h5py.File(path, "r")

        self.trajectories = []
        primitive_names = trajectories

        if primitive_names is None:
            primitive_names = list(self.file.keys())

        for primitive_name in primitive_names:
            primitive_group = self.file[primitive_name]
            for traj_name in primitive_group.keys():
                traj_group = primitive_group[traj_name]
                self.trajectories.append({
                    "primitive": primitive_name,
                    "name": traj_name,
                    "pose": traj_group["pose"][:],
                    "action": traj_group["action"][:],
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

        
        self.train_objective = train_objective
        self.trajectory_indices = list(range(len(self.trajectories)))  # index into self.trajectories
        self.active_trajectory_indices = self.trajectory_indices.copy()

    def __len__(self):
        if self.train_objective == "autoregressive":
            return len(self.transitions)
        elif self.train_objective == "flow_matching":
            return len(self.trajectory_indices)

    def __getitem__(self, idx):

        if self.train_objective == "autoregressive": 

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

                "start_pose": torch.from_numpy(self.normalize_pose(traj["pose"][0])).float(),
                "goal_pose": torch.from_numpy(self.normalize_pose(traj["pose"][-1])).float(),

                "frame_idx": frame_idx,
                "trajectory_length": T,
                "trajectory_t": torch.tensor([trajectory_t], dtype=torch.float32),

                "delta_pose": torch.from_numpy(delta_pose).float(),
                "action": torch.from_numpy(action),

                "trajectory": traj["name"]
            }
        elif self.train_objective == "flow_matching": 
            traj_idx = self.trajectory_indices[idx]
            return self.get_trajectory(traj_idx)
        else: 
            print("Unknown training objective")
            raise RuntimeError
        
    def get_trajectory(self, idx):
        traj = self.trajectories[idx]
        return {

            "trajectory": torch.from_numpy(
                self.normalize_pose(traj["pose"])
            ).float(),

            "trajectory_raw": torch.from_numpy(
                traj["pose"]
            ).float(),

            "start_pose": torch.from_numpy(
                self.normalize_pose(traj["pose"][0])
            ).float(),

            "goal_pose": torch.from_numpy(
                self.normalize_pose(traj["pose"][-1])
            ).float(),

            "action": torch.from_numpy(
                traj["action"]
            ).float(),

            "primitive": traj["primitive"],
            "name": traj["name"],
        }
    
    def set_active_primitives(self, trajectories=None):

        if trajectories is None:
            active = range(len(self.trajectories))
        elif isinstance(trajectories, str):
            active = [
                i for i, traj in enumerate(self.trajectories)
                if traj["primitive"] == trajectories
            ]
        else:
            active = [
                i for i, traj in enumerate(self.trajectories)
                if traj["primitive"] in trajectories
            ]

        self.transitions = []
        for traj_idx in active:
            T = len(self.trajectories[traj_idx]["pose"])
            for frame in range(T-1):
                self.transitions.append(
                    (traj_idx, frame)
                )

        self.trajectory_indices = list(active)
        self.active_trajectory_indices = list(active)


    def set_active_trajectory(self, idx):
        self.trajectory_indices = [idx]
        self.transitions = []
        T = len(self.trajectories[idx]["pose"])
        for frame in range(T-1):
            self.transitions.append((idx, frame))

        self.active_trajectory_indices = [idx]
    
    @property
    def num_trajectories(self):
        return len(self.trajectories)
    
    @property
    def trajectory_names(self):
        return [
            f'{t["primitive"]}/{t["name"]}'
            for t in self.trajectories
        ]
    
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