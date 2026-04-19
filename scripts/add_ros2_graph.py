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

import shutil  # noqa: E402
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

ROBOT_PATH = "/volcaniarm"
CAMERA_LINK_PATH = f"{ROBOT_PATH}/camera_link"
CAMERA_SENSOR_PATH = f"{CAMERA_LINK_PATH}/camera_sensor"
CMD_TOPIC = "/joint_commands"
STATE_TOPIC = "/joint_states"
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

    # Clone the closed-loop USD to our output path, then open the COPY
    # in context so every authoring call (pxr + og.Controller) lands in
    # the same root layer and is persisted on Save(). This sidesteps
    # session-layer and sublayer-composition edge cases.
    shutil.copy(BASE_USD, OUT_USD)
    ctx = omni.usd.get_context()
    ctx.open_stage(str(OUT_USD))
    stage = ctx.get_stage()
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))
    _app.update()
    print("[add_ros2_graph] opened copy", flush=True)

    robot_prim = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(f"Expected prim {ROBOT_PATH} not found in {OUT_USD.name}")
    stage.SetDefaultPrim(robot_prim)

    UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.CreateSizeAttr(50.0)
    UsdGeom.XformCommonAPI(ground).SetTranslate(Gf.Vec3d(0.0, 0.0, GROUND_Z - 25.0))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    print("[add_ros2_graph] ground + robot ready", flush=True)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
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
        {"graph_path": "/ROS2Graph", "evaluator_name": "execution"},
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
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
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
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_PATH)]),
                ("SubscribeJointState.inputs:topicName", CMD_TOPIC),
                ("ArticulationController.inputs:robotPath", ROBOT_PATH),
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

    # Dump what's actually under /ROS2Graph right now (pre-save).
    print("[add_ros2_graph] /ROS2Graph tree on stage:")
    graph_root = stage.GetPrimAtPath("/ROS2Graph")
    if graph_root and graph_root.IsValid():
        for p in Usd.PrimRange(graph_root):
            print(f"  {p.GetPath()}  ({p.GetTypeName()})")
    else:
        print("  (missing — graph was not authored)")

    stage.GetRootLayer().Save()

    # Re-open the saved file in a separate stage to verify persistence.
    verify = Usd.Stage.Open(str(OUT_USD))
    verify_root = verify.GetPrimAtPath("/ROS2Graph")
    print("[add_ros2_graph] /ROS2Graph tree in saved file:")
    if verify_root and verify_root.IsValid():
        for p in Usd.PrimRange(verify_root):
            print(f"  {p.GetPath()}  ({p.GetTypeName()})")
    else:
        print("  (missing — Save() did not persist the graph)")
    print(f"Wrote: {OUT_USD}")


if __name__ == "__main__":
    main()
    _app.close()
