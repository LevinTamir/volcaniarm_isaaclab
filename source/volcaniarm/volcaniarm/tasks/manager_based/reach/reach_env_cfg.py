# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Reach task for the volcaniarm 5-bar planar 2-DOF arm.

Policy observes (actuated + passive joint state, EE pose command, last
action) and outputs position targets for the two elbow joints. Reward is
L2 distance from `left_ee_link` to a sampled target pose, plus a tanh
shaping term and small action/velocity penalties. No orientation reward
— the arm is planar and can't track orientation independently of
position.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .volcaniarm_cfg import VOLCANIARM_CFG


##
# Scene definition
##


@configclass
class VolcaniarmReachSceneCfg(InteractiveSceneCfg):
    """Scene: ground plane, dome light, volcaniarm."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot = VOLCANIARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Target pose sampled uniformly in a YZ-plane box in front of the arm."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_ee_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Targets are in `base_link` frame. The arm is planar —
            # left_ee_link X is fixed at +0.071 (kinematically), so we
            # pin target X there. Y is 20 cm inside the cart's inner
            # leg rails (legs at y=±0.698). Z spans world 0→0.2 m
            # (cart base sits at world z=0.98, so base-frame z = world-0.98):
            # the weed-detection use case has targets at ground level.
            pos_x=(0.071, 0.071),
            pos_y=(-0.50, 0.50),
            pos_z=(-0.98, -0.78),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


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
    @configclass
    class PolicyCfg(ObsGroup):
        # Actuated elbow joints only — the real ros2_control hardware
        # exposes only these via encoders. Joint velocity is omitted
        # because the hardware derives it by finite-differencing position
        # (volcaniarm_hardware.cpp), which is too noisy / laggy to match
        # Isaac's clean velocity at deployment. `last_action` already
        # carries recent motion intent.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["volcaniarm_(left|right)_elbow_joint"])},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset: small uniform offset on the two elbow joints."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["volcaniarm_(left|right)_elbow_joint"]),
            # Offset range narrowed to ±π/4 (45°) so reset stays inside the
            # mechanical limits: default ± π/4 = [π/4, 3π/4] (right) and
            # [-3π/4, -π/4] (left). Both fall within the new [π/4, 8π/9]
            # and [-8π/9, -π/4] joint limits with margin.
            "position_range": (-0.7853981633974483, 0.7853981633974483),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    # === Goal 1: EE reaches the desired location (PRIMARY) ===
    #
    # Weights heavily boosted — reach is the #1 task and must dominate
    # the penalty terms. Max positive contribution at target ≈ +4.0/step
    # (+1440/episode), out-signaling any realistic penalty combination.
    # L2 distance for far-range gradient, broad tanh for workspace-wide
    # attraction, fine tanh for precision near the target.
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_tanh_broad = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"]),
            "std": 0.3,
            "command_name": "ee_pose",
        },
    )
    end_effector_position_tracking_tanh_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ee_link"]),
            "std": 0.05,
            "command_name": "ee_pose",
        },
    )

    # === Goal 2: Avoid elbow-down assembly mode ===
    #
    # Rewards mean knee Z above the base — both 5-bar assembly modes
    # reach the same EE pose, so tracking alone is ambiguous between
    # them. This term breaks the tie toward elbow-up.
    elbow_up = RewTerm(
        func=mdp.elbow_up_posture,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["volcaniarm_(left|right)_arm_link"]),
        },
    )

    # === Goal 3: Avoid self-collision ===
    #
    # Sums max(0, 0.15 - ||com_a - com_b||) over the four cross-side
    # link pairs. Zero when every pair is ≥ 15 cm apart at the COM;
    # grows as any pair approaches. Threshold tunable — bump to 0.20
    # if you see collisions, drop to 0.10 if it's blocking reach.
    link_proximity = RewTerm(
        func=mdp.link_pair_proximity,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pairs": [
                ("volcaniarm_left_elbow_link", "volcaniarm_right_elbow_link"),
                ("volcaniarm_left_elbow_link", "volcaniarm_right_arm_link"),
                ("volcaniarm_left_arm_link", "volcaniarm_right_elbow_link"),
                ("volcaniarm_left_arm_link", "volcaniarm_right_arm_link"),
            ],
            "min_distance": 0.15,
        },
    )

    # === Minor support (smoothness) ===
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- Previously tried, disabled now ---
    # Redundant with hard URDF limits on actuated joints + closure physics:
    # arm_joint_in_range = RewTerm(
    #     func=mdp.joint_pos_out_of_range, weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot",
    #         joint_names=["volcaniarm_(left|right)_arm_joint"]),
    #         "low": -1.5707963267948966, "high": 1.5707963267948966})
    # closure_violation = RewTerm(
    #     func=mdp.closure_body_distance, weight=-5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")})
    # Superseded by link_proximity (general) and elbow_up:
    # arm_crossover = RewTerm(
    #     func=mdp.arm_crossover, weight=-0.3,
    #     params={"asset_cfg": SceneEntityCfg("robot")})
    # Early working-range attempt (fought with reach):
    # joint_pos_in_range = RewTerm(
    #     func=mdp.joint_pos_out_of_range, weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot",
    #         joint_names=["volcaniarm_(left|right)_elbow_joint"]),
    #         "low": -1.047, "high": 1.047})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class VolcaniarmReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: VolcaniarmReachSceneCfg = VolcaniarmReachSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
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
class VolcaniarmReachEnvCfg_PLAY(VolcaniarmReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
