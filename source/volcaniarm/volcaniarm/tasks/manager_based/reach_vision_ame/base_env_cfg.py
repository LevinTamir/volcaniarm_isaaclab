# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Base scene/env cfgs shared by the AME vision reach task.

Migrated verbatim from the retired `reach_vision` task (the ResNet18
baseline that AME superseded): the scene (robot, camera rig, lights,
ground), actions, rewards and terminations carried over unchanged; the
old task's observations/events did not (AME has its own).

The class names keep the `ReachVision` spelling so checkpoints and logs
that reference cfg class paths stay readable.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..reach.volcaniarm_cfg import VOLCANIARM_CFG
from . import mdp

# Workspace bounds in `volcaniarm_base_link` frame (base sits at world
# z=0.98). Identical to the validated state-based task ranges.
WEED_X_BASE = 0.071
WEED_Y_RANGE = (-0.50, 0.50)
WEED_Z_RANGE = (-0.98, -0.78)

# Joint-range reward bounds, MIRRORED left/right — measured 2026-07-22 by
# joystick-driving the sim arm around the intended envelope and recording
# /joint_states extents (ws scripts/record_joint_extents.py), then
# symmetrising across the mirror pairs, +~3 deg margin, rounded to 5 deg.
# Each elbow flexes further inward (70 deg) than outward (45 deg); the arms
# mirror likewise — which is why the old single shared arm bound
# (-90..+50 deg for BOTH arms) never fit: the parked pose itself violated it.
ELBOW_IN_RAD = 1.2217304763960306  # 70 deg (inward flex)
ELBOW_OUT_RAD = 0.7853981633974483  # 45 deg (outward flex)
ARM_BIG_RAD = 1.4835298641951802  # 85 deg
ARM_SMALL_RAD = 0.5235987755982988  # 30 deg

# Image resolution — small enough that 512+ envs fit on one GPU,
# large enough for ResNet18 ImageNet weights to be in distribution
# (model expects ~224x224 but tolerates smaller; we accept that
# tradeoff for throughput).
IMG_W, IMG_H = 96, 96

FLOOR_COLOR = (0.72, 0.70, 0.68)  # matches scripts/build_lab.py


# ---------------------------------------------------------------------
# Camera placement — edit this block to retune the rig
# ---------------------------------------------------------------------
# Conceptually the same as the URDF's `camera_mount_linear_joint` (xyz)
# and `camera_mount_rev_joint` (rpy): expose translation + Euler angles
# directly so they can be tweaked without touching quaternion math.
#
# CAMERA_PARENT_PRIM is the prim the offset is applied relative to. Two
# good choices:
#   "{ENV_REGEX_NS}/Robot/camera_link"
#       Mirrors `volcaniarm_ros2.usd` — the camera lives on the same
#       table-mount chain as the real RealSense (camera_mount_linear →
#       camera_mount_rev → camera_link). Default below; xyz/rpy are
#       expressed in the URDF's `camera_link` body frame (X forward,
#       Y left, Z up).
#   "{ENV_REGEX_NS}/Robot/volcaniarm_base_link"
#       For a "camera fixed on the robot base looking down" rig — bypass
#       the table mount entirely. Reset xyz/rpy below to taste.
#
# CAMERA_OFFSET_RPY is roll/pitch/yaw in **radians**, applied as an
# extrinsic XYZ Euler rotation (URDF convention). Positive roll is
# right-handed about X, etc.
#
# The default rpy here is (-π/2, 0, -π/2) — the standard
# `camera_link` → `camera_link_optical` transform. Combined with
# convention="ros" in the TiledCamera spec, the rendered RGB lines up
# with `camera_color_optical_frame` from the ROS2 bridge, so a policy
# trained here matches what `/camera/color/image_raw` looks like at
# deploy time.
CAMERA_PARENT_PRIM = "{ENV_REGEX_NS}/Robot/camera_link"
CAMERA_OFFSET_XYZ = (0.0, 0.0, 0.0)
CAMERA_OFFSET_RPY = (-math.pi / 2.0, 0.0, -math.pi / 2.0)


def _quat_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Extrinsic XYZ Euler (rad) → (w, x, y, z) quaternion."""
    r, p, y = rpy
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


CAMERA_OFFSET_QUAT = _quat_from_rpy(CAMERA_OFFSET_RPY)

##
# Scene definition
##


@configclass
class VolcaniarmReachVisionSceneCfg(InteractiveSceneCfg):
    """Scene: concrete-grey ground, dome light, robot, weed, camera."""

    # Concrete-grey floor. `GroundPlaneCfg` and `TerrainImporterCfg`
    # both spawn the default Isaac grid USD (whose base texture is
    # blue), which would dominate the rendered image even after a
    # diffuse_color override. Spawn an explicit colored cuboid 1 cm
    # thick at z<0 instead — same pattern `scripts/build_lab.py` uses
    # for the lab world. The cuboid is treated as a static collider
    # via `collision_props`; no rigid body so it doesn't fall.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.MeshCuboidCfg(
            size=(40.0, 40.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=FLOOR_COLOR),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )
    robot = VOLCANIARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Dome alone leaves the floor dim and tinted by the default sky.
    # A weaker dome + a dimmer top-down distant light gives even,
    # true-coloured rendering — bright enough that diffuse_color
    # shows through, dim enough that the floor isn't washed out
    # toward white. Calibrated so a (0.72, 0.70, 0.68) floor reads
    # ~(180, 175, 170) in the rendered PNG.
    dome = AssetBaseCfg(
        prim_path="/World/dome",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=400.0),
    )
    sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 0.95),
            intensity=1200.0,
            angle=10.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.7071, 0.7071, 0.0, 0.0)),
    )

    # Weed surrogate: green cylinder. Disabled gravity so it stays
    # exactly where reset_root_state_uniform places it — we don't
    # care about weed dynamics, only its position as a target.
    weed = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Weed",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.10,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        # Init pose is overwritten on every reset by `randomize_weed_pose`.
        # World z below: workspace centre = 0.98 + (-0.88) = 0.10.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(WEED_X_BASE, 0.0, 0.10)),
    )

    # Camera placement is driven by the CAMERA_* constants at the top
    # of this file — edit those, not this block. Intrinsics mirror
    # `scripts/add_ros2_graph.py` so the train-time rig matches the
    # ROS2 USD verbatim.
    base_camera = TiledCameraCfg(
        prim_path=f"{CAMERA_PARENT_PRIM}/base_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_XYZ,
            rot=CAMERA_OFFSET_QUAT,
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14721,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.05, 50.0),
        ),
        width=IMG_W,
        height=IMG_H,
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Position targets for the two actuated elbow joints."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["volcaniarm_(left|right)_elbow_joint"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Apr-22-style stack adapted to weed Y-Z tracking (X ignored)."""

    weed_position_tracking = RewTerm(
        func=mdp.position_weed_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"])},
    )
    weed_position_tracking_tanh_broad = RewTerm(
        func=mdp.position_weed_error_tanh,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"]),
            "std": 0.3,
        },
    )
    weed_position_tracking_tanh_fine = RewTerm(
        func=mdp.position_weed_error_tanh,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"]),
            "std": 0.05,
        },
    )
    # Four per-joint terms because the bounds are mirrored, not shared.
    left_elbow_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["volcaniarm_left_elbow_joint"]),
            "low": -ELBOW_IN_RAD,
            "high": ELBOW_OUT_RAD,
        },
    )
    right_elbow_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["volcaniarm_right_elbow_joint"]),
            "low": -ELBOW_OUT_RAD,
            "high": ELBOW_IN_RAD,
        },
    )
    left_arm_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["volcaniarm_left_arm_joint"]),
            "low": -ARM_SMALL_RAD,
            "high": ARM_BIG_RAD,
        },
    )
    right_arm_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["volcaniarm_right_arm_joint"]),
            "low": -ARM_BIG_RAD,
            "high": ARM_SMALL_RAD,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##

@configclass
class VolcaniarmReachVisionEnvCfg(ManagerBasedRLEnvCfg):
    # Image rendering is the bottleneck — drop env count vs the
    # state-based task's 4096.
    scene: VolcaniarmReachVisionSceneCfg = VolcaniarmReachVisionSceneCfg(
        num_envs=512, env_spacing=2.5
    )
    # observations/events are deliberately NOT set here: the retired
    # reach_vision task's ResNet18 observations and its EventCfg were not
    # migrated. The AME subclass supplies its own (AmeObservationsCfg /
    # AmeEventCfg).
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0
        # Wide diagonal view over the env grid (2.5 m spacing): frames a
        # ~3x3 block of robots in the training videos instead of one.
        self.viewer.eye = (10.0, 10.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.9)
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

