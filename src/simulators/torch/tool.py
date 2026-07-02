import math
import cv2 
import numpy as np

class Tool:

    def __init__(self, trajectory):

        self.trajectory = trajectory
        self.step_idx = 0

        self.x, self.y = trajectory.get_position(0)
        self.theta = 0.0
        self.omega = 0.1

        self.vertices = [
            (-40, -20),
            (40, -20),
            (40, 20),
            (-40, 20)
        ]

    def step(self):

        self.step_idx += 1

        if self.step_idx >= self.trajectory.steps:
            return

        self.x, self.y = self.trajectory.get_position(self.step_idx)
        self.theta += self.omega

    def contains(self, point):

        verts = np.array(self.get_world_vertices(),dtype=np.float32)

        ### Return whether the mass node at 'point' is inside the polygon defined by 'verts' 
        result = cv2.pointPolygonTest(verts,(float(point[0]), float(point[1])),False)

        return result >= 0
    
    def get_pose(self):

        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta
        }
    
    def get_world_vertices(self):

        verts = []

        c = math.cos(self.theta)
        s = math.sin(self.theta)

        ### Rotate every vertex around the center: 
        # [cos(theta) - sin(theta)] * [x]
        # [sin(theta)   cos(theta)] * [y]
        for vx, vy in self.vertices:
            x_rot = c * vx - s * vy
            y_rot = s * vx + c * vy
            verts.append((
                self.x + x_rot,
                self.y + y_rot
            ))

        return verts
        

class LinearTrajectory:
    def __init__(self,start_x,start_y,dx,dy,steps):

        self.start_x = start_x
        self.start_y = start_y

        self.dx = dx
        self.dy = dy

        self.steps = steps

    def get_position(self, step):
        return (self.start_x + step * self.dx,self.start_y + step * self.dy)
    