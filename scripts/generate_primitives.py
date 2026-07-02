
import numpy as np

import numpy as np


class PrimitivesGenerator:

    def __init__(self, H=256, W=256):

        self.H = H
        self.W = W

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def compute_action(self, pose):

        action = np.zeros_like(pose)

        action[:-1] = pose[1:] - pose[:-1]
        action[-1] = action[-2]

        return action

    def random_position(self, margin=40):

        x = np.random.uniform(margin, self.W - margin)
        y = np.random.uniform(margin, self.H - margin)

        return x, y

    # --------------------------------------------------
    # Primitive 1
    # --------------------------------------------------

    def generate_line(self, T):

        print("Generating Line Primitive")

        pose = np.zeros((T,3), dtype=np.float32)

        angle = np.random.uniform(0,2*np.pi)
        speed = np.random.uniform(1,3)

        print("Angle: ", angle)
        print("Speed: ", speed) 

        dx = speed*np.cos(angle)
        dy = speed*np.sin(angle)

        print("dx: ", dx)
        print("dy: ", dy)

        x,y = self.random_position()

        theta = angle
        
        for t in range(T):
            pose[t] = [x,y,theta]
            x += dx
            y += dy

        return pose

    # --------------------------------------------------
    # Primitive 2
    # --------------------------------------------------

    def generate_circle(self,T):

        pose = np.zeros((T,3),dtype=np.float32)
        radius = np.random.uniform(60, 90)

        cx = np.random.uniform(radius+20,self.W-radius-20)
        cy = np.random.uniform(radius+20,self.H-radius-20)

        direction = np.random.choice([-1,1])
        omega = direction*np.random.uniform(0.03,0.08)
        angle = np.random.uniform(0,2*np.pi)

        for t in range(T):

            x = cx + radius*np.cos(angle)
            y = cy + radius*np.sin(angle)

            theta = angle + direction*np.pi/2

            pose[t] = [x,y,theta]

            angle += omega

        return pose

    # --------------------------------------------------
    # Primitive 3
    # --------------------------------------------------

    def generate_arc(self,T):

        pose = np.zeros((T,3),dtype=np.float32)

        radius = np.random.uniform(60, 90)

        cx = np.random.uniform(radius+20,self.W-radius-20)
        cy = np.random.uniform(radius+20,self.H-radius-20)

        direction = np.random.choice([-1,1])

        angle = np.random.uniform(0,2*np.pi)

        total_angle = np.random.uniform(np.pi/3,np.pi)

        omega = direction*total_angle/(T-1)

        for t in range(T):

            x = cx + radius*np.cos(angle)
            y = cy + radius*np.sin(angle)

            theta = angle + direction*np.pi/2

            pose[t] = [x,y,theta]

            angle += omega

        return pose

    # --------------------------------------------------
    # Primitive 4
    # --------------------------------------------------

    def generate_rectangle(self,T):

        pose = np.zeros((T,3),dtype=np.float32)

        w = np.random.uniform(80, 160)
        h = np.random.uniform(80, 160)

        x0 = np.random.uniform(40,self.W-w-40)
        y0 = np.random.uniform(40,self.H-h-40)

        corners = np.array([
            [x0,y0],
            [x0+w,y0],
            [x0+w,y0+h],
            [x0,y0+h],
            [x0,y0]
        ])

        pts = T//4

        idx = 0

        for edge in range(4):

            p0 = corners[edge]
            p1 = corners[edge+1]

            theta = np.arctan2(
                p1[1]-p0[1],
                p1[0]-p0[0]
            )

            for i in range(pts):

                alpha = i/pts

                p = (1-alpha)*p0 + alpha*p1

                pose[idx] = [p[0],p[1],theta]

                idx += 1

        pose[idx:] = pose[idx-1]

        return pose

    # --------------------------------------------------
    # Primitive 5
    # --------------------------------------------------

    def generate_zigzag(self,T):

        pose = np.zeros((T,3),dtype=np.float32)

        x,y = self.random_position()

        theta = np.pi/4

        speed = 2

        segment = 20

        for t in range(T):

            pose[t] = [x,y,theta]

            x += speed*np.cos(theta)
            y += speed*np.sin(theta)

            if (t+1)%segment==0:

                theta *= -1

        return pose

    # --------------------------------------------------
    # Primitive 6
    # --------------------------------------------------

    def generate_figure8(self,T):

        pose = np.zeros((T,3),dtype=np.float32)

        cx = self.W/2
        cy = self.H/2

        a = np.random.uniform(70, 100)

        ts = np.linspace(0,2*np.pi,T)

        for i,t in enumerate(ts):

            x = cx + a*np.sin(t)
            y = cy + a*np.sin(t)*np.cos(t)

            if i<T-1:

                dt = ts[i+1]-t

                dx = a*np.cos(t)

                dy = a*(np.cos(t)**2 - np.sin(t)**2)

                theta = np.arctan2(dy,dx)

            else:

                theta = pose[i-1,2]

            pose[i] = [x,y,theta]

        return pose

    # --------------------------------------------------

    def generate_random(self,T):

        generators = [
            ("line", self.generate_line),
            ("circle", self.generate_circle),
            ("arc", self.generate_arc),
            ("rectangle", self.generate_rectangle),
            ("zigzag", self.generate_zigzag),
            ("figure8", self.generate_figure8),
        ]

        primitive, generator = generators[
            np.random.randint(len(generators))
        ]

        pose = generator(T)

        action = self.compute_action(pose)

        return pose, action, primitive