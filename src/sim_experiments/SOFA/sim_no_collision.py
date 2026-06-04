import Sofa
import Sofa.Core
import numpy as np

''' WORKS BUT NO COLLISION '''

class ToolController(Sofa.Core.Controller):

    def __init__(self, tool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool = tool
        self.counter = 0

    def onAnimateBeginEvent(self, event):

        pose = self.tool.dofs.position.value.copy()

        # move down slowly
        pose[0][0] = 1.0 * np.sin(0.01*self.counter)
        pose[0][1] -= 0.005

        self.tool.dofs.position.value = pose

        print("Tool Y:", pose[0][1])


def createScene(root):

    # ==================================================
    # GLOBAL SETTINGS
    # ==================================================

    root.dt = 0.01

    # disable gravity for now
    root.gravity = [0, 0, 0]

    root.addObject(
        "RequiredPlugin",
        pluginName=[
            "Sofa.Component.ODESolver.Backward",
            "Sofa.Component.LinearSolver.Iterative",
            "Sofa.Component.Mass",
            "Sofa.Component.StateContainer",
            "Sofa.Component.Topology.Container.Grid",
            "Sofa.Component.SolidMechanics.Spring",
            "Sofa.Component.Constraint.Projective",
            "Sofa.Component.Mapping.Linear",
            "Sofa.Component.IO.Mesh",
            "Sofa.GL.Component.Rendering3D"
        ]
    )

    root.addObject("DefaultAnimationLoop")

    root.addObject(
        "VisualStyle",
        displayFlags="showBehaviorModels showForceFields showWireframe"
    )

    root.addObject("EulerImplicitSolver")
    root.addObject("CGLinearSolver")

    # ==================================================
    # CLOTH
    # ==================================================

    cloth = root.addChild("cloth")

    cloth.addObject(
        "RegularGridTopology",
        name="grid",
        nx=5,
        ny=5,
        nz=1,
        xmin=-2,
        xmax=2,
        ymin=0,
        ymax=4,
        zmin=0,
        zmax=0
    )

    cloth.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        showObject=True,
        showObjectScale=0.3
    )

    cloth.addObject(
        "UniformMass",
        totalMass=1.0
    )

    cloth.addObject(
        "RegularGridSpringForceField",
        stiffness=50,
        damping=1
    )

    cloth.addObject(
        "FixedProjectiveConstraint",
        indices=[20, 21, 22, 23, 24]
    )

    # ==================================================
    # TOOL
    # ==================================================

    tool = root.addChild("tool")

    tool.addObject(
        "MechanicalObject",
        name="dofs",
        template="Rigid3d",
        position=[0, 5, 0, 0, 0, 0, 1],
        showObject=False
    )

    root.addObject(
        ToolController(
            tool=tool,
            name="toolController"
        )
    )

    # ==================================================
    # VISUAL MODEL
    # ==================================================

    visual = tool.addChild("visual")

    visual.addObject(
        "MeshOBJLoader",
        name="loader",
        filename="/home/aaron-schulze/SOFA/v25.12.00/plugins/InfinyToolkit/share/sofa/examples/InfinyToolkit/mesh/sphere.obj"
    )

    visual.addObject(
        "OglModel",
        src="@loader",
        scale3d=[0.5, 0.5, 0.5],
        color=[1, 0, 0, 1]
    )

    visual.addObject(
        "RigidMapping"
    )

    return root