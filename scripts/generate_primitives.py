
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
    
    def trajectory_in_bounds(self, pose, margin=20):
        x = pose[:, 0]
        y = pose[:, 1]

        return (
            x.min() >= margin and
            x.max() <= self.W - margin and
            y.min() >= margin and
            y.max() <= self.H - margin
        )

    # --------------------------------------------------
    # Primitive 1
    # --------------------------------------------------

    def generate_line(self, T):

        margin = 20

        while True:

            angle = np.random.uniform(0, 2 * np.pi)

            path_length = np.random.uniform(40, 180)

            x0, y0 = self.random_position(margin=margin)

            x1 = x0 + path_length * np.cos(angle)
            y1 = y0 + path_length * np.sin(angle)

            # accept only valid trajectories
            if (
                margin <= x1 <= self.W - margin and
                margin <= y1 <= self.H - margin
            ):
                break

        pose = np.zeros((T, 3), dtype=np.float32)

        alpha = np.linspace(0, 1, T)

        pose[:, 0] = x0 + alpha * (x1 - x0)
        pose[:, 1] = y0 + alpha * (y1 - y0)
        pose[:, 2] = angle

        return pose

    # --------------------------------------------------
    # Primitive 2
    # --------------------------------------------------

    def random_circle_center(self, radius, margin=20):

        cx = np.random.uniform(
            radius + margin,
            self.W - radius - margin,
        )

        cy = np.random.uniform(
            radius + margin,
            self.H - radius - margin,
        )

        return cx, cy

    def generate_circle(self,T):

        pose = np.zeros((T,3),dtype=np.float32)
        radius = np.random.uniform(25, 90)

        cx, cy = self.random_circle_center(radius)

        direction = np.random.choice([-1,1])
        num_turns = np.random.uniform(0.5, 2.0)

        omega = direction * (2 * np.pi * num_turns) / (T - 1)
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

        radius = np.random.uniform(25, 90)

        cx, cy = self.random_circle_center(radius)

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

    def generate_rectangle(self, T):

        margin = 20

        while True:

            pose = np.zeros((T, 3), dtype=np.float32)

            width = np.random.uniform(50, 150)
            height = np.random.uniform(50, 150)

            center_x, center_y = self.random_position(margin=margin + max(width, height) / 2)

            angle = np.random.uniform(0, 2 * np.pi)

            direction = np.random.choice([-1, 1])

            start_corner = np.random.randint(4)

            # rectangle in local coordinates
            corners = np.array([
                [-width / 2, -height / 2],
                [ width / 2, -height / 2],
                [ width / 2,  height / 2],
                [-width / 2,  height / 2],
            ])

            # rotate rectangle
            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)],
            ])

            corners = corners @ R.T
            corners += np.array([center_x, center_y])

            # choose traversal direction
            if direction == -1:
                corners = corners[::-1]

            # random starting corner
            corners = np.roll(corners, -start_corner, axis=0)

            # close loop
            corners = np.vstack([corners, corners[0]])

            pts_per_edge = T // 4

            idx = 0

            for edge in range(4):

                p0 = corners[edge]
                p1 = corners[edge + 1]

                theta = np.arctan2(
                    p1[1] - p0[1],
                    p1[0] - p0[0],
                )

                for i in range(pts_per_edge):

                    alpha = i / pts_per_edge

                    p = (1 - alpha) * p0 + alpha * p1

                    pose[idx] = [p[0], p[1], theta]

                    idx += 1

            pose[idx:] = pose[idx - 1]

            if self.trajectory_in_bounds(pose):
                return pose
    # --------------------------------------------------
    # Primitive 5
    # --------------------------------------------------

    def generate_zigzag(self, T):

        margin = 20

        while True:

            pose = np.zeros((T, 3), dtype=np.float32)

            x, y = self.random_position(margin=margin)

            base_angle = np.random.uniform(0, 2 * np.pi)

            deviation = np.random.uniform(np.pi / 8, np.pi / 3)

            speed = np.random.uniform(1.0, 3.0)

            segment = np.random.randint(8, 20)

            theta = base_angle + deviation

            sign = 1

            for t in range(T):

                pose[t] = [x, y, theta]

                x += speed * np.cos(theta)
                y += speed * np.sin(theta)

                if (t + 1) % segment == 0:

                    sign *= -1

                    theta = base_angle + sign * deviation

            if self.trajectory_in_bounds(pose):
                return pose

    # --------------------------------------------------
    # Primitive 6
    # --------------------------------------------------

    def generate_figure8(self, T):

        margin = 20

        while True:

            pose = np.zeros((T, 3), dtype=np.float32)

            # Size of the figure-8
            a = np.random.uniform(30, 70)

            # Random center (large enough so the entire curve fits)
            cx = np.random.uniform(
                margin + a,
                self.W - margin - a,
            )

            cy = np.random.uniform(
                margin + a / 2,
                self.H - margin - a / 2,
            )

            # Random global orientation
            rotation = np.random.uniform(0, 2 * np.pi)

            ts = np.linspace(0, 2 * np.pi, T)

            R = np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation),  np.cos(rotation)],
            ])

            for i, t in enumerate(ts):

                # Local figure-8 (lemniscate-like)
                p = np.array([
                    a * np.sin(t),
                    a * np.sin(t) * np.cos(t),
                ])

                # Rotate
                p = R @ p

                x = cx + p[0]
                y = cy + p[1]

                # Tangent in local coordinates
                dp = np.array([
                    a * np.cos(t),
                    a * (np.cos(t)**2 - np.sin(t)**2),
                ])

                dp = R @ dp

                theta = np.arctan2(dp[1], dp[0])

                pose[i] = [x, y, theta]

            if self.trajectory_in_bounds(pose):
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