import cv2
import numpy as np
import math
import os


def render(
    pose,
    polygon,
    tissue=None,
    gt_pose=None,
    canvas_size=(800, 800),
):
    """
    Parameters
    ----------
    pose : (3,)
        (x, y, theta) of the tool to render.

    polygon : (N,2)
        Tool vertices in LOCAL coordinates.

    tissue : optional
        Tissue object.

    gt_pose : optional
        Ground-truth pose for overlay visualization.

    canvas_size : (H,W)
    """

    H, W = canvas_size

    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # -------------------------------------------------------
    # Draw tissue (optional)
    # -------------------------------------------------------

    if tissue is not None:

        pos = tissue.pos.cpu().numpy()

        for i, j in tissue.springs:

            cv2.line(
                canvas,
                tuple(pos[i].astype(int)),
                tuple(pos[j].astype(int)),
                (0, 0, 0),
                1,
            )

        for idx, p in enumerate(pos):

            color = (0, 0, 255)
            radius = 4

            if idx in tissue.contact_nodes:
                color = (0, 255, 0)
                radius = 7

            cv2.circle(
                canvas,
                tuple(p.astype(int)),
                radius,
                color,
                -1,
            )

    # -------------------------------------------------------
    # Draw predicted tool
    # -------------------------------------------------------

    draw_polygon(
        canvas,
        pose,
        polygon,
        color=(255, 0, 0),
    )

    # -------------------------------------------------------
    # Draw GT tool
    # -------------------------------------------------------

    if gt_pose is not None:

        draw_polygon(
            canvas,
            gt_pose,
            polygon,
            color=(0, 255, 0),
        )

    # -------------------------------------------------------
    # Text
    # -------------------------------------------------------

    x, y, theta = pose

    cv2.putText(
        canvas,
        f"x={x:.1f}  y={y:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    cv2.putText(
        canvas,
        f"theta={math.degrees(theta):.1f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    return canvas

def draw_polygon(
    canvas,
    pose,
    local_vertices,
    color,
):

    scale = 3 

    x, y, theta = pose
    x *= scale
    y *= scale


    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    verts = local_vertices * scale 
    verts = verts @ R.T
    verts += np.array([x, y])

    verts = verts.astype(np.int32)

    cv2.fillPoly(
        canvas,
        [verts],
        color,
    )



def render_rollout(predicted, gt, output_dir, filename):

    tool_polygon = np.array([
        [-20,-6],
        [20,-6],
        [20,6],
        [-20,6],
    ],dtype=np.float32)

    writer = cv2.VideoWriter(
        os.path.join(output_dir, filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (800,800),
    )

    for t in range(len(predicted)):
        img = render(
            pose=predicted[t],
            gt_pose=gt[t] if gt != None else None,
            polygon=tool_polygon,
        )
        writer.write(img)
    writer.release()

    print(f"Saved {output_dir}/{filename}")

