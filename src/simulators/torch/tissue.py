import torch

class Tissue:

    def __init__(self,rows=20,cols=20,spacing=30,device="cpu"):

        self.device = device
        self.rows=rows
        self.cols=cols
        positions = []

        self.contact_nodes = []

        ### Define mass nodes in a grid
        for r in range(rows):
            for c in range(cols):
                positions.append([c * spacing,r * spacing])
        
        self.pos = torch.tensor(positions, dtype=torch.float32, device=device)
        self.initial_pos = self.pos.clone()

        self.vel = torch.zeros_like(self.pos)

        ### Define springs: horizontal, vertical and diagonal (stored in a 1D array)
        self.springs = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c < cols - 1:                                #   * - *
                    right = idx + 1
                    self.springs.append((idx, right))
                if r < rows - 1:                                #   * 
                    below = idx + cols                          #   |
                    self.springs.append((idx, below))           #   *

                if r < rows - 1 and c < cols - 1:               #   * 
                    self.springs.append((idx, idx + cols + 1))  #    \
                                                                #     *

                if r < rows - 1 and c > 0:                      #     *
                    self.springs.append((idx, idx + cols - 1))  #    /
                                                                #   *

        self.rest_lengths = []

        for i, j in self.springs:
            length = torch.norm(self.pos[i] - self.pos[j])
            self.rest_lengths.append(length)

        self.rest_lengths = torch.tensor(self.rest_lengths,device=device)

    def compute_forces(self,k=60.0,damping=10):

        forces = torch.zeros_like(self.pos)

        for spring_idx, (i, j) in enumerate(self.springs):

            p1 = self.pos[i]
            p2 = self.pos[j]

            delta = p2 - p1
            length = torch.norm(delta)          # current spring length

            if length < 1e-6:
                continue

            direction = delta / length          # unit vector of length 1 (spring direction)
            stretch = (length - self.rest_lengths[spring_idx])      # (x-x0) = "by how much spring got stretched"
            force = k * stretch * direction                         # Hooke's law

            v_rel = self.vel[j] - self.vel[i]
            damping_force = (                       # Project velocity onto spring direction
                damping                             # Oppose relative motion
                * torch.dot(v_rel, direction)
                * direction
            )

            force += damping_force

            forces[i] += force                      # Equal and opposite foces
            forces[j] -= force

        forces += -0.1 * self.vel                   # Global damping

        return forces
    
    def step(self, tool, dt=0.01):

        forces = self.compute_forces()
        contact_forces, self.contact_nodes = self.compute_contact_forces(tool)
        forces += contact_forces

        acc = forces

        self.vel += acc * dt
        self.pos += self.vel * dt

        self.pin_top_row()

    def get_deformation_field(self):
        return self.pos - self.initial_pos

    def pin_top_row(self):
        self.pos[:self.cols] = self.initial_pos[:self.cols]
        self.vel[:self.cols] = 0

    def compute_contact_forces(self, tool):

        forces = torch.zeros_like(self.pos)
        contact_nodes = []

        for i in range(len(self.pos)):
            if tool.contains(self.pos[i]):
                forces[i,1] += 2000
                contact_nodes.append(i)

        return forces, contact_nodes