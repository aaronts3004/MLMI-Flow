import torch.nn as nn
import torch

class PoseMLP(nn.Module):

    def __init__(self, pose_dim=3):
        super().__init__()

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
