import cv2

from tissue import Tissue
from tool import Tool, LinearTrajectory
from renderer import render


tissue = Tissue()
traj = LinearTrajectory(280,0,0,15,500)
tool = Tool(traj)

trajectory_dataset = []
frame_idx = 0
while True:

    frame_idx += 1 

    tool.step()
    tissue.step(tool)
    img = render(tissue, tool)

    trajectory_dataset.append({
        "frame": frame_idx, 
        "tool_pose": tool.get_pose(),
        "deformation": tissue.get_deformation_field()
    })

    cv2.imshow("Mass Spring",img)
    key = cv2.waitKey(10)
    if key == 27:
        break