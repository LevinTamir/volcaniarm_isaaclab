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


##
# Scene definition
##


@configclass
class VolcaniarmReachVisionSceneCfg(InteractiveSceneCfg):
    """Scene: concrete-grey ground, dome light, robot, plant, camera."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=FLOOR_COLOR),
    )
    robot = VOLCANIARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
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

    # Downward-looking camera mounted on the robot's base link.
    # Offset is in base_link frame; the same (xyz, rpy) will be reused
    # in the URDF when Phase B (ROS2 deployment) starts — robot link
    # tree is shared CAD between USD and URDF.
    #
    # Position: 0.30 m above the base, centred on the workspace X. The
    # base sits at world z=0.98 → camera at world z≈1.28 looks straight
    # down onto the workspace (world z 0–0.20). With FOV ~70°, horizontal
    # coverage at the floor is ~1.5 m — well over the Y range ±0.5.
    #
    # Rotation: 180° around X (quat w,x,y,z = 0,1,0,0) flips the
    # camera's "ros" forward axis (+Z) to point in -base_z = -world_z
    # = downward. The image's vertical axis is then aligned with the
    # robot's Y axis.
    base_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/volcaniarm_base_link/base_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(PLANT_X_BASE, 0.0, 0.30),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
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
    """Reset events: arm joints + plant pose + plant colour."""

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
