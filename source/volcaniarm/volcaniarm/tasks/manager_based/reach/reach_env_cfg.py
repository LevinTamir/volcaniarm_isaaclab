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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
            # pin target X there. Y spans the cart's inner leg rails
            # (legs at y=±0.698). Z covers the weed-reach band: weeds
            # grow 20-30 cm tall from the ground, so the EE needs to
            # reach roughly 15-30 cm above ground (world z 0.15→0.30 m).
            # Base sits at world z=0.98 → base-frame z = world-0.98.
            pos_x=(0.071, 0.071),
            pos_y=(-0.45, 0.45),
            pos_z=(-0.83, -0.68),
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
            # Reset offset kept inside the ±65° (±1.1345 rad) mechanical
            # limits with ~14° of margin on each side.
            "position_range": (-0.9, 0.9),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    # Baseline reward: EE position tracking + a soft penalty for elbow
    # joint excursions beyond the measured ±65° mechanical range.
    # URDF keeps wide limits so PhysX's closure-constraint solver stays
    # stable — we teach the policy to respect the working range via the
    # soft penalty instead.
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
    # Actuated elbow joints: ±75° working range.
    elbow_pos_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["volcaniarm_(left|right)_elbow_joint"]),
            "low": -1.3089969389957472,   # -75°
            "high": 1.3089969389957472,   # +75°
        },
    )
    # Passive arm_joints: range widened to [-100°, +60°] (was [-90°, +50°]).
    # The reach-up workspace requires passives near the edges; the original
    # band was too tight and the penalty kept fighting reach.
    arm_pos_in_range = RewTerm(
        func=mdp.joint_pos_out_of_range,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["volcaniarm_(left|right)_arm_joint"]),
            "low": -1.7453292519943295,   # -100°
            "high": 1.0471975511965976,   # +60°
        },
    )

    # Smoothness penalties — tiny weights, kill the jittery motion seen
    # in play.py without competing with reach.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    # Linearly expand the EE target box over the first ~60% of training:
    #   iter 0-100: tiny center box (y±10cm, z=-0.78 only) — easy reach
    #   iter 100-300: linear expansion in both axes
    #   iter 300+: full weed-reach workspace (y±45cm, z [-0.83, -0.68])
    expand_targets = CurrTerm(
        func=mdp.expand_ee_target_box,
        params={
            "command_name": "ee_pose",
            "start_iter": 100,
            "end_iter": 300,
            "y_start": 0.10,
            "y_end": 0.45,
            "z_start": (-0.78, -0.78),
            "z_end": (-0.83, -0.68),
            "num_steps_per_env": 32,
        },
    )


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
    curriculum: CurriculumCfg = CurriculumCfg()
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
