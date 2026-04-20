"""Author the ROS2 overlay for the volcaniarm.

Reads:   assets/usd/volcaniarm_closed.usd   (closed-loop articulation)
Writes:  assets/usd/volcaniarm_ros2.usd     (sublayers closed + adds ROS2 graph)

The overlay adds:
  * ground plane at z = -0.98 m (cart wheels rest on it)
  * dome light
  * RealSense-ish Camera prim parented to `/volcaniarm/camera_link`
  * OmniGraph `/ROS2Graph`:
      - publishes  /clock                        (rosgraph_msgs/Clock)
      - publishes  /joint_states                 (sensor_msgs/JointState, 4 articulated joints)
      - subscribes /joint_commands               (sensor_msgs/JointState) → ArticulationController
      - publishes  /camera/color/image_raw       (sensor_msgs/Image, rgb)
      - publishes  /camera/color/camera_info     (sensor_msgs/CameraInfo)
      - publishes  /camera/depth/color/points    (sensor_msgs/PointCloud2, colored depth)

After generating, open the USD in Isaac Sim GUI and press Play. Test from
a ROS2 terminal:
    ros2 topic echo /joint_states
    ros2 topic pub -1 /joint_commands sensor_msgs/msg/JointState \\
      "{name: [volcaniarm_left_elbow_joint, volcaniarm_right_elbow_joint], \\
        position: [0.5, -0.5]}"
    ros2 topic hz /camera/color/image_raw
    ros2 topic hz /camera/depth/color/points

Run with (outside any conda env — uses Isaac Sim's bundled Python so
the ROS2 bridge's rclpy version matches and the full kit loads):
    conda deactivate  # if needed
    ~/isaac/isaac-sim/python.sh scripts/add_ros2_graph.py
"""

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})
print("[add_ros2_graph] SimulationApp up", flush=True)

from pathlib import Path  # noqa: E402

from isaacsim.core.utils import extensions  # noqa: E402

extensions.enable_extension("isaacsim.ros2.bridge")
_app.update()

import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
import usdrt.Sdf  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
BASE_USD = PROJECT / "assets/usd/volcaniarm_closed.usd"
OUT_USD = PROJECT / "assets/usd/volcaniarm_ros2.usd"

WORLD_PATH = "/World"
ROBOT_PATH = f"{WORLD_PATH}/volcaniarm"
# The URDF converter applies PhysicsArticulationRootAPI to `root_joint`
# (not the wrapping Xform). ArticulationController's targetPrim has to
# point there exactly.
ARTICULATION_ROOT_PATH = f"{ROBOT_PATH}/root_joint"
CAMERA_LINK_PATH = f"{ROBOT_PATH}/camera_link"
CAMERA_SENSOR_PATH = f"{CAMERA_LINK_PATH}/camera_sensor"
GRAPH_PATH = f"{WORLD_PATH}/ROS2Graph"
CMD_TOPIC = "/joint_commands"
# Isaac publishes the raw articulation state (all 4 joints) to a separate topic.
# The ros2_control TopicBasedSystem on the ROS side consumes this; /joint_states
# stays owned by joint_state_broadcaster (elbows) + ClosedLoopTrajectoryController
# (passive arm + closure joints) to match the Gazebo topology.
STATE_TOPIC = "/isaac_joint_states"
CLOCK_TOPIC = "/clock"
CAM_RGB_TOPIC = "color/image_raw"
CAM_INFO_TOPIC = "color/camera_info"
CAM_PCL_TOPIC = "depth/color/points"
CAM_NAMESPACE = "camera"
CAM_FRAME = "camera_color_optical_frame"
CAM_WIDTH = 640
CAM_HEIGHT = 480
GROUND_Z = -0.98  # matches -min(z) of cart + caster wheel meshes


def main() -> None:
    if not BASE_USD.exists():
        raise FileNotFoundError(f"Run close_loop.py first — missing {BASE_USD}")
    if OUT_USD.exists():
        OUT_USD.unlink()

    # Author a fresh USD with /World as the default prim and everything
    # (robot-as-reference, ground, light, camera, graph) nested under
    # /World. That way the scene comes through identically whether the
    # file is opened directly, passed on the command line, or added as
    # a reference in a wrapping scene.
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))
    print("[add_ros2_graph] new stage", flush=True)

    world = UsdGeom.Xform.Define(stage, WORLD_PATH)
    stage.SetDefaultPrim(world.GetPrim())

    robot_xform = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    robot_xform.GetPrim().GetReferences().AddReference(str(BASE_USD))
    _app.update()
    if not stage.GetPrimAtPath(ROBOT_PATH).IsValid():
        raise RuntimeError(f"Reference to {BASE_USD.name} did not resolve at {ROBOT_PATH}")
    # NOTE: don't apply ArticulationRootAPI here — the reference already
    # carries it through from the base USD. Applying it again triggers
    # `UsdPhysics: Nested articulation roots are not allowed`.
    print("[add_ros2_graph] robot referenced under /World/volcaniarm", flush=True)

    # The URDF converter baked stiffness=400/damping=40 into every
    # joint drive. For the 5-bar's *passive* arm joints that's wrong —
    # they're not stepper-driven, and the spring-back-to-zero drive
    # fights the closure constraint (causes ~10 rad/s oscillation at
    # rest). Zero them out here; elbows keep the active drive.
    _passive_joints = {"volcaniarm_left_arm_joint", "volcaniarm_right_arm_joint"}
    for prim in stage.Traverse():
        if prim.GetName() in _passive_joints and prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.1)
    print("[add_ros2_graph] passive arm joint drives zeroed", flush=True)

    ground = UsdGeom.Cube.Define(stage, f"{WORLD_PATH}/GroundPlane")
    ground.CreateSizeAttr(50.0)
    UsdGeom.XformCommonAPI(ground).SetTranslate(Gf.Vec3d(0.0, 0.0, GROUND_Z - 25.0))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    print("[add_ros2_graph] ground ready", flush=True)

    dome = UsdLux.DomeLight.Define(stage, f"{WORLD_PATH}/DomeLight")
    dome.CreateIntensityAttr(2500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

    if not stage.GetPrimAtPath(CAMERA_LINK_PATH).IsValid():
        raise RuntimeError(
            f"Expected camera_link prim at {CAMERA_LINK_PATH} — the URDF must have the "
            "camera_link + camera_mount_rev_link chain; regenerate convert_urdf.py if missing."
        )
    camera = UsdGeom.Camera.Define(stage, CAMERA_SENSOR_PATH)
    camera.CreateFocalLengthAttr(18.14721)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 50.0))
    print("[add_ros2_graph] camera prim defined", flush=True)

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CameraHelperRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("CameraHelperPCL", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                # jointNames is NOT connected to the subscribe node — it's
                # hard-coded below to the two elbow joints. We also only
                # wire positionCommand (no velocity/effort); steppers on
                # the real arm are position-controlled.
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperRGB.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperInfo.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperPCL.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperRGB.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperInfo.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperPCL.inputs:renderProductPath"),
                ("Context.outputs:context", "CameraHelperRGB.inputs:context"),
                ("Context.outputs:context", "CameraHelperInfo.inputs:context"),
                ("Context.outputs:context", "CameraHelperPCL.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", CLOCK_TOPIC),
                ("PublishJointState.inputs:topicName", STATE_TOPIC),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ARTICULATION_ROOT_PATH)]),
                ("SubscribeJointState.inputs:topicName", CMD_TOPIC),
                ("ArticulationController.inputs:targetPrim", [usdrt.Sdf.Path(ARTICULATION_ROOT_PATH)]),
                (
                    "ArticulationController.inputs:jointNames",
                    ["volcaniarm_left_elbow_joint", "volcaniarm_right_elbow_joint"],
                ),
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_SENSOR_PATH)]),
                ("CreateRenderProduct.inputs:width", CAM_WIDTH),
                ("CreateRenderProduct.inputs:height", CAM_HEIGHT),
                ("CameraHelperRGB.inputs:type", "rgb"),
                ("CameraHelperRGB.inputs:topicName", CAM_RGB_TOPIC),
                ("CameraHelperRGB.inputs:frameId", CAM_FRAME),
                ("CameraHelperRGB.inputs:nodeNamespace", CAM_NAMESPACE),
                ("CameraHelperInfo.inputs:topicName", CAM_INFO_TOPIC),
                ("CameraHelperInfo.inputs:frameId", CAM_FRAME),
                ("CameraHelperInfo.inputs:nodeNamespace", CAM_NAMESPACE),
                ("CameraHelperPCL.inputs:type", "depth_pcl"),
                ("CameraHelperPCL.inputs:topicName", CAM_PCL_TOPIC),
                ("CameraHelperPCL.inputs:frameId", CAM_FRAME),
                ("CameraHelperPCL.inputs:nodeNamespace", CAM_NAMESPACE),
            ],
        },
    )

    print("[add_ros2_graph] graph built", flush=True)

    stage.GetRootLayer().Export(str(OUT_USD))

    verify = Usd.Stage.Open(str(OUT_USD))
    verify_graph = verify.GetPrimAtPath(GRAPH_PATH)
    n_nodes = sum(1 for _ in Usd.PrimRange(verify_graph)) - 1 if verify_graph and verify_graph.IsValid() else 0
    print(f"[add_ros2_graph] saved file has {GRAPH_PATH} with {n_nodes} child nodes", flush=True)
    print(f"Wrote: {OUT_USD}")


if __name__ == "__main__":
    main()
    _app.close()
