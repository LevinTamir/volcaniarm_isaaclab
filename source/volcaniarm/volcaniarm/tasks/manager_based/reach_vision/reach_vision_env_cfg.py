# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Vision reach task for the volcaniarm 5-bar planar 2-DOF arm.

Differs from `reach`:
- A green-cylinder plant surrogate is spawned per env at a random
  (Y, Z) inside the existing reachable workspace (X is fixed by the
  planar mechanism).
- A downward-looking TiledCamera mounted on `volcaniarm_base_link`
  produces RGB at the training resolution.
- Asymmetric observations: actor sees ResNet18 features + joint_pos_rel
  + last_action; critic adds the privileged plant position in base
  frame.
- Reward is Y-Z distance from `left_ee_link` to the plant root_pos_w
  (X error ignored — planar mechanism cannot control X).
- Floor uses concrete-grey colour matching `scripts/build_lab.py`
  FLOOR_COLOR (0.72, 0.70, 0.68) so the visual gap to the real lab
  floor is small.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..reach.volcaniarm_cfg import VOLCANIARM_CFG
from . import mdp


# Workspace bounds in `volcaniarm_base_link` frame (base sits at world
# z=0.98). Identical to the validated state-based task ranges.
PLANT_X_BASE = 0.071
PLANT_Y_RANGE = (-0.50, 0.50)
PLANT_Z_RANGE = (-0.98, -0.78)

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
    """Scene: concrete-grey ground, dome light, robot, plant, camera."""

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

    # Plant surrogate: green cylinder. Disabled gravity so it stays
    # exactly where reset_root_state_uniform places it — we don't
    # care about plant dynamics, only its position as a target.
    plant = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plant",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.10,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        # Init pose is overwritten on every reset by `randomize_plant_pose`.
        # World z below: workspace centre = 0.98 + (-0.88) = 0.10.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(PLANT_X_BASE, 0.0, 0.10)),
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
class ObservationsCfg:
    """Asymmetric obs: actor=features+proprio, critic adds plant pose.

    rsl_rl's ActorCritic concatenates all groups inside the "policy"
    and "critic" lists; both must be flat 1D per-term so the MLP can
    ingest. ResNet18 features are 1000-dim (ImageNet logits) — that's
    what the IsaacLab stdlib's `image_features` returns when called
    with model_name="resnet18".
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["volcaniarm_(left|right)_elbow_joint"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        image_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["volcaniarm_(left|right)_elbow_joint"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        image_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        plant_pos_b = ObsTerm(func=mdp.plant_pos_in_base)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Reset + interval events: joint reset, plant pose, lighting, colour, camera DR."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["volcaniarm_(left|right)_elbow_joint"]
            ),
            "position_range": (-0.9, 0.9),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Plant: random (Y, Z) inside the workspace; X pinned at the
    # planar-arm reachable plane. Pose is in WORLD frame, so add the
    # base-link world z (0.98) to the base-frame Z range.
    randomize_plant_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("plant"),
            "pose_range": {
                "x": (PLANT_X_BASE, PLANT_X_BASE),
                "y": PLANT_Y_RANGE,
                "z": (0.98 + PLANT_Z_RANGE[0], 0.98 + PLANT_Z_RANGE[1]),
            },
            "velocity_range": {},
        },
    )

    # Mid-episode plant resample — gives ~3 reach attempts per 12 s episode
    # instead of 1, tripling the on-policy reaching signal per env per iter.
    resample_plant_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg("plant"),
            "pose_range": {
                "x": (PLANT_X_BASE, PLANT_X_BASE),
                "y": PLANT_Y_RANGE,
                "z": (0.98 + PLANT_Z_RANGE[0], 0.98 + PLANT_Z_RANGE[1]),
            },
            "velocity_range": {},
        },
    )

    # Floor color jitter — small variation around the concrete-grey base.
    # Global asset (one prim shared across envs); the change applies to all
    # envs from the next render onward.
    randomize_floor_color = EventTerm(
        func=mdp.randomize_visual_color_global,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ground"),
            "base_color": FLOOR_COLOR,
            "variation": 0.05,
        },
    )

    # Dome light: ±25% intensity, ±10% balanced color drift.
    randomize_dome = EventTerm(
        func=mdp.randomize_light,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dome"),
            "intensity_range": (320.0, 480.0),
            "color_base": (0.75, 0.75, 0.75),
            "color_variation": 0.10,
        },
    )

    # Sun light: ±25% intensity, ±10% balanced color drift (warm bias kept).
    randomize_sun = EventTerm(
        func=mdp.randomize_light,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sun"),
            "intensity_range": (900.0, 1500.0),
            "color_base": (1.0, 1.0, 0.95),
            "color_variation": 0.10,
        },
    )

    # Camera pose jitter — per-env, additive on each reset. Small std
    # values keep the accumulated drift bounded over a 3000-iter run
    # (drift ≈ σ·√N_resets; with σ_pos=1 mm, σ_rpy≈0.17°, that's ~2 cm
    # and ~3° by training end — within plausible mounting tolerance.)
    randomize_camera_pose = EventTerm(
        func=mdp.randomize_camera_pose,
        mode="reset",
        params={
            "sensor_name": "base_camera",
            "pos_std": (0.001, 0.001, 0.001),
            "rpy_std": (0.003, 0.003, 0.003),
        },
    )


@configclass
class RewardsCfg:
    """Apr-22-style stack adapted to plant Y-Z tracking (X ignored)."""

    plant_position_tracking = RewTerm(
        func=mdp.position_plant_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"])},
    )
    plant_position_tracking_tanh_broad = RewTerm(
        func=mdp.position_plant_error_tanh,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"]),
            "std": 0.3,
        },
    )
    plant_position_tracking_tanh_fine = RewTerm(
        func=mdp.position_plant_error_tanh,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"]),
            "std": 0.05,
        },
    )
    elbow_pos_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["volcaniarm_(left|right)_elbow_joint"]
            ),
            "low": -1.3089969389957472,
            "high": 1.3089969389957472,
        },
    )
    arm_pos_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["volcaniarm_(left|right)_arm_joint"]
            ),
            "low": -1.5707963267948966,
            "high": 0.8726646259971648,
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
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation


@configclass
class VolcaniarmReachVisionEnvCfg_PLAY(VolcaniarmReachVisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
