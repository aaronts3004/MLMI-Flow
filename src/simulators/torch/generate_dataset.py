import cv2
import h5py
import numpy as np

# ------------------------
# Parameters
# ------------------------

H = 256
W = 256

T = 100

tool_w = 40
tool_h = 12

dx = 2
dy = 0
dtheta = 0

# ------------------------
# Allocate
# ------------------------

rgb = np.zeros((1, T, H, W, 3), dtype=np.uint8)
pose = np.zeros((1, T, 3), dtype=np.float32)
action = np.zeros((1, T, 3), dtype=np.float32)

# ------------------------
# Initial pose
# ------------------------

x = 50.
y = 128.
theta = 0.

# ------------------------
# Rollout
# ------------------------

for t in range(T):

    pose[0,t] = [x,y,theta]
    action[0,t] = [dx,dy,dtheta]

    img = np.ones((H,W,3),dtype=np.uint8)*255

    rect = (
        (x,y),
        (tool_w,tool_h),
        theta
    )

    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    cv2.fillPoly(img,[box],(40,40,220))

    rgb[0,t]=img

    x += dx
    y += dy
    theta += dtheta

# ------------------------
# Save
# ------------------------

with h5py.File("rigid_dataset.h5","w") as f:

    f.create_dataset("rgb",data=rgb)
    f.create_dataset("pose",data=pose)
    f.create_dataset("action",data=action)