import cv2
import numpy as np
import math 

def render(tissue, tool):

    ### Main canvas
    canvas = np.ones((800,800,3), dtype=np.uint8) * 255
    pos = tissue.pos.cpu().numpy()

    ### Spring = segment between each mass node
    for i,j in tissue.springs:

        p1 = tuple(pos[i].astype(int))
        p2 = tuple(pos[j].astype(int))

        cv2.line(canvas, p1, p2, (0,0,0), 1)

    ### Tool = Filled polygon
    verts = np.array(
        tool.get_world_vertices(),
        dtype=np.int32
    )

    cv2.fillPoly(canvas, [verts], (255,0,0))

    ### Mass node = circle
    for idx, p in enumerate(pos):
        color = (0,0,255)   # red
        radius = 4

        if idx in tissue.contact_nodes:
            color = (0, 255,0)   # green
            radius = 7

        cv2.circle(
            canvas,
            tuple(p.astype(int)),
            radius,
            color,
            -1
        )

    cv2.putText(
        canvas,
        f"x={tool.x:.1f} y={tool.y:.1f}",
        (580, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,0,0),
        2
    )

    cv2.putText(
        canvas,
        f"theta={math.degrees(tool.theta):.1f} deg",
        (580, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,0,0),
        2
    )

    
    return canvas