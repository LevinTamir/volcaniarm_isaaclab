# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""AME-style vision reach task: green mask + CNN + spatial attention.

Differs from `reach_vision`:

- **Scene matches the real lab**: matte black rubber mat instead of concrete
  grey, and the actual 3D-printed weed mesh instead of a green cylinder.
- **Weed stands on the mat** at a measured, reachable position. Its USD origin
  is at the canopy apex (see `scripts/convert_weed.py`), so `root_pos_w` is the
  reach target and the mesh hangs down to z=0. The arm cannot reach the mat
  itself — the EE floor is z=0.052 m — so the canopy is the only sensible
  target. The weed is 11 cm (1.5x the authored STL), which clears that floor
  by ~6 cm and gives a measured Y span of about +-0.10 m.
- **Observations are three groups, not two.** The image is no longer squeezed
  through a frozen ResNet18 into 1000 ImageNet logits; instead the raw camera
  frame is reduced to a green-coverage map that the policy's own CNN consumes.
  `policy` carries proprio only, `mask` carries the map, `critic` carries
  proprio plus the privileged weed pose. rsl_rl hands these to the module as a
  TensorDict keyed by group, so nothing has to slice a flat vector.

Rewards, actions, terminations and the lighting/camera randomisation are
reused verbatim from `reach_vision` so the head-to-head comparison is honest.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..reach_vision.reach_vision_env_cfg import (
    ActionsCfg,
    RewardsCfg,
    TerminationsCfg,
    VolcaniarmReachVisionEnvCfg,
    VolcaniarmReachVisionSceneCfg,
)
from . import mdp
from .contract import (
    MASK_GROUP,
    MASK_H,
    MASK_W,
    MAT_COLOR,
    MAT_ROUGHNESS,
    WEED_COLOR,
    WEED_SPAWN_Z_WORLD,
    WEED_X_BASE,
    WEED_Y_RANGE,
    WEED_Z_JITTER,
)

ELBOW_JOINTS = ["volcaniarm_(left|right)_elbow_joint"]

def _project_root() -> Path:
    """Walk up from this file to the repo root (the dir holding `assets/`).

    Searching beats a fixed `parents[N]` index: the package sits six levels
    down (source/volcaniarm/volcaniarm/tasks/manager_based/<task>/), which is
    easy to miscount and fails as a silent FileNotFoundError at scene build.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "assets" / "usd").is_dir():
            return parent
    raise RuntimeError("could not locate project root containing assets/usd")


# Built by scripts/convert_weed.py. Resolved relative to this file so the task
# works regardless of the process CWD.
WEED_USD = str(_project_root() / "assets" / "usd" / "fake_weed.usd")


##
# Scene
##


@configclass
class VolcaniarmReachVisionAmeSceneCfg(VolcaniarmReachVisionSceneCfg):
    """Real-lab scene: black mat + printed weed mesh. Robot/camera/lights inherited."""

    # Matte black rubber mat. Kept as a thin cuboid rather than GroundPlaneCfg
    # for the same reason as the base task: the stock ground USD carries a blue
    # grid texture that survives a diffuse_color override.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.MeshCuboidCfg(
            size=(40.0, 40.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=MAT_COLOR, roughness=MAT_ROUGHNESS, metallic=0.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    # The printed weed. Visual-only (no collider, gravity off) — it is a reach
    # *target*, never contacted, and we want it to stay exactly where the reset
    # event puts it. The material is bound here rather than baked into the USD
    # so colour randomisation later is a config change, not a re-convert.
    weed = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Weed",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WEED_USD,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=WEED_COLOR, roughness=0.6, metallic=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        # Deliberately the origin. `reset_root_state_uniform` computes
        # `default_root_state + env_origin + sample`, so the pose_range is an
        # OFFSET, not an absolute position — putting the spawn pose here as
        # well would double-count it (and did: the weed landed at 2x both
        # X and Z). Keeping this at zero makes the event's ranges read as
        # absolute env-local coordinates, which is what they look like.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


##
# MDP
##


@configclass
class AmeObservationsCfg:
    """Three groups: proprio, mask, and the privileged critic vector.

    The mask lives in its own group rather than being concatenated into
    `policy` for two reasons: rsl_rl 3.x passes observations as a TensorDict
    keyed by group, so the module can read `obs["mask"]` without any index
    arithmetic; and a group referenced by several algorithmic sets is computed
    once per step instead of once per consumer (the old task ran ResNet18
    twice — once for the policy group, once for the critic group).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ORDER IS LOAD-BEARING: must equal contract.PROPRIO_TERMS, because the
        # ONNX bundle concatenates its inputs in this order. The exporter
        # asserts it.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ELBOW_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class MaskCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.green_mask,
            # Small additive noise then clipping keeps the policy from keying
            # on exact coverage values, which differ between the renderer and a
            # real sensor even when the hue matches.
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(0.0, 1.0),
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "out_hw": (MASK_H, MASK_W),
                "isolate": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ELBOW_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)
        # Privileged: the exact target. This is why the critic needs no image —
        # a vision encoder on top of the ground-truth position would only add
        # cost and make the value gradient compete for the shared conv weights.
        weed_pos_b = ObsTerm(
            func=mdp.weed_pos_in_base, params={"weed_cfg": SceneEntityCfg("weed")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    mask: MaskCfg = MaskCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class AmeEventCfg:
    """Reset events. Lighting/camera DR inherited in spirit from `reach_vision`.

    The weed is sampled along a *line*: X is pinned by the planar mechanism, Z
    is pinned because the weed stands on the mat, and only Y varies. The Y
    range is measured (see `scripts/check_workspace.py`), not inherited — the
    older tasks sampled +-0.50 while only [-0.287, 0.299] is reachable.
    """

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ELBOW_JOINTS),
            "position_range": (-0.9, 0.9),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_weed_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("weed"),
            "pose_range": {
                "x": (WEED_X_BASE, WEED_X_BASE),
                "y": WEED_Y_RANGE,
                "z": (
                    WEED_SPAWN_Z_WORLD + WEED_Z_JITTER[0],
                    WEED_SPAWN_Z_WORLD + WEED_Z_JITTER[1],
                ),
            },
            "velocity_range": {},
        },
    )

    # Mid-episode resample: ~3 reach attempts per 12 s episode instead of 1.
    resample_weed_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg("weed"),
            "pose_range": {
                "x": (WEED_X_BASE, WEED_X_BASE),
                "y": WEED_Y_RANGE,
                "z": (
                    WEED_SPAWN_Z_WORLD + WEED_Z_JITTER[0],
                    WEED_SPAWN_Z_WORLD + WEED_Z_JITTER[1],
                ),
            },
            "velocity_range": {},
        },
    )

    # Mat colour jitter. Tiny, because the mask keys on saturation and hue and
    # the mat has neither — this is here for visual realism, not signal.
    randomize_mat_color = EventTerm(
        func=mdp.randomize_visual_color_global,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ground"),
            "base_color": MAT_COLOR,
            "variation": 0.02,
        },
    )

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

    randomize_camera_pose = EventTerm(
        func=mdp.randomize_camera_pose,
        mode="reset",
        params={
            "sensor_name": "base_camera",
            "pos_std": (0.001, 0.001, 0.001),
            "rpy_std": (0.003, 0.003, 0.003),
        },
    )


##
# Environment
##


@configclass
class VolcaniarmReachVisionAmeEnvCfg(VolcaniarmReachVisionEnvCfg):
    scene: VolcaniarmReachVisionAmeSceneCfg = VolcaniarmReachVisionAmeSceneCfg(
        num_envs=512, env_spacing=2.5
    )
    observations: AmeObservationsCfg = AmeObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: AmeEventCfg = AmeEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Nothing to retarget: the inherited reward terms resolve their target
        # through `weed_cfg`, which already defaults to SceneEntityCfg("weed"),
        # and this scene names its target entity `weed`.


@configclass
class VolcaniarmReachVisionAmeEnvCfg_PLAY(VolcaniarmReachVisionAmeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.mask.enable_corruption = False
