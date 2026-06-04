import taichi as ti
import numpy as np
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)


# *** Mass-Spring grid definition *** 
n = 64
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)
x = ti.Vector.field(2, dtype=float, shape=(n, n))
v = ti.Vector.field(2, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(2, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

# *** Definition of forces and sim parameters ! in 2D ! ***
gravity = ti.Vector([0.0, 0.0])
spring_Y = 3000
dashpot_damping = 500
drag_damping = 5
bending_springs = False
contact_Y = 5000                ### contact stiffness

# *** Rigid tool definition *** 
tool_radius = 0.15
tool_center = ti.Vector.field(
    2,
    dtype=float,
    shape=()
)
tool_center[None] = [-0.4, 0.4]
tool_theta = 0.0 
tool_W = 0.09
tool_H = 0.09
center = tool_center[None]

tool_render = np.array(tool_center[None])

num_contacts = 0

@ti.kernel
def initialize_mass_points():

    print("Initializing mass points!!")

    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5,
            j * quad_size - 0.5
        ]
        v[i, j] = [0.0, 0.0]

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()

                if current_dist >= 1e-8:

                    d = x_ij / current_dist

                    stretch = current_dist - original_dist
                    force += -spring_Y * stretch * d
                    force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v[i] += force * dt

    for i in ti.grouped(x):

        if i[0] == 0 or i[0] == n - 1 or i[1] == 0 or i[1] == n - 1:
            v[i] = ti.Vector([0.0, 0.0])
            continue

        v[i] *= ti.exp(-drag_damping * dt)

        rel = x[i] - tool_center[None]
        c = ti.cos(tool_theta)
        s = ti.sin(tool_theta)

        local = ti.Vector([
            c*rel[0] + s*rel[1],
            -s*rel[0] + c*rel[1]
        ])

        inside = (
            abs(local[0]) < tool_W/2 and
            abs(local[1]) < tool_H/2
        )
        if inside:

            px = tool_W/2 - abs(local[0])
            py = tool_H/2 - abs(local[1])
            penetration = min(px,py)
            normal = ti.Vector([
                1.0 if local[0] > 0 else -1.0,
                1.0 if local[1] > 0 else -1.0
            ])
            contact_force = (
                contact_Y *
                penetration *
                normal
            )
        
            v[i] += contact_force * dt

        x[i] += dt * v[i]

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

@ti.kernel
def copy_positions_to_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


# *** MAIN LOOP AND GUI ***
gui = ti.GUI("2D Tissue Prototype", (1000, 1000))

print(x.to_numpy().min())
print(x.to_numpy().max())

initialize_mass_points()
print(x[0,0])
print(x[n-1,n-1])

t = 0

while gui.running:

    for _ in range(substeps):
        substep()

    copy_positions_to_vertices()

    gui.clear(0xFFFFFF)

    pos = vertices.to_numpy()

    print(
        pos[:,0].min(),
        pos[:,0].max(),
        pos[:,1].min(),
        pos[:,1].max()
    )

    pos[:,0] = 0.5 + pos[:,0] * 0.8
    pos[:,1] = 0.5 + pos[:,1] * 0.8

    gui.circles(
        pos,
        radius=2,
        color=0x0066AA
    )

    for i in range(n):
        for j in range(n-1):
            gui.line(
                pos[i*n+j],
                pos[i*n+j+1],
                color=0x888888
            )

    for i in range(n-1):
        for j in range(n):
            gui.line(
                pos[i*n+j],
                pos[(i+1)*n+j],
                color=0x888888
            )

    corners = np.array([
        [-tool_W/2, -tool_H/2],
        [ tool_W/2, -tool_H/2],
        [ tool_W/2,  tool_H/2],
        [-tool_W/2,  tool_H/2]
    ])

    c = np.cos(tool_theta)
    s = np.sin(tool_theta)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    world_corners = (
        corners @ R.T
        + np.array(tool_center[None])
    )

    render_corners = world_corners.copy()

    render_corners[:,0] = 0.5 + 0.8 * render_corners[:,0]
    render_corners[:,1] = 0.5 + 0.8 * render_corners[:,1]

    for k in range(4):
        gui.line(
            render_corners[k],
            render_corners[(k+1)%4],
            radius=2,
            color=0xFF0000
        )

    tool_theta += 0.01
    tool_center[None][0] = tool_center[None][0] + 0.004 * t 
    tool_center[None][1] = tool_center[None][1] - 0.001 * t
    t += 0.01

    gui.show()