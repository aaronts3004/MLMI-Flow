import torch.nn as nn
import torch

class PoseMLP(nn.Module):

    def __init__(self, pose_dim=3):
        super().__init__()

        print("Instantiating new PoseMLP")

        self.net = nn.Sequential(
            nn.Linear(pose_dim + 1,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,pose_dim)
        )

    def forward(self, pose, time):

        x = torch.cat([pose, time], dim=-1)
        return self.net(x)


class TrajectoryMLP(nn.Module):

    def __init__(
        self,
        trajectory_length=100,
        pose_dim=3,
        hidden_dim=512,
    ):
        super().__init__()
        print("Instantiating new TrajectoryMLP")
        self.T = trajectory_length
        self.pose_dim = pose_dim

        input_dim = trajectory_length * pose_dim + 1
        output_dim = trajectory_length * pose_dim

        self.linear = nn.Linear(301,300)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, trajectory, tau):

        B = trajectory.shape[0]

        trajectory = trajectory.reshape(B, -1)

        tau = tau.reshape(B, 1)

        x = torch.cat([trajectory, tau], dim=-1)

        out = self.linear(x)

        return out.reshape(B, self.T, self.pose_dim)