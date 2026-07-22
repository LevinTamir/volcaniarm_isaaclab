# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 fine-tuning env: same task, heavier domain randomization.

Paper-style two-stage curriculum: stage 1 (`...AmeEnvCfg`, task id `-v0`)
learns the reach under mild DR; stage 2 (this cfg, task id `-v1`) resumes
from a stage-1 checkpoint and hardens the policy against camera-pose error,
color/threshold miscalibration and missed detections. Observation dims,
rewards, actions and terminations are identical to stage 1, so
`OnPolicyRunner.load()` accepts a stage-1 checkpoint unchanged.

The stage-2 constants live HERE, not in `contract.py`: they are sim-only
training knobs. The deploy contract (GREEN_NOMINAL, mask dims, proprio
order) is untouched by this stage — the ONNX export bakes the same nominal
thresholds either way.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .contract import WEED_COLOR
from .reach_vision_ame_env_cfg import AmeEventCfg, VolcaniarmReachVisionAmeEnvCfg

# Wider HSV threshold jitter than contract.GREEN_JITTER — the per-env band
# the mask keys on wanders further from nominal, so the policy stops trusting
# exact coverage values. Keep hue_center + hue_halfwidth excursions inside
# the weed's actual hue or an unlucky env goes blind for its whole episode.
GREEN_JITTER_STAGE2 = dict(
    hue_center=0.060,
    hue_halfwidth=0.050,
    sat_min=0.120,
    val_min=0.120,
)

# Mask dropout: patches emulate partial occlusion punching holes in the
# detection (applied pre-isolate_blob, so fragments compete exactly like at
# deploy); whole-frame drops emulate stale/black frames.
MASK_DROPOUT_STAGE2 = dict(
    patch_prob=0.3,
    num_patches=(1, 3),
    patch_size=(4, 12),
    frame_drop_prob=0.05,
)


@configclass
class Stage2EventCfg(AmeEventCfg):
    """Stage-1 events with the DR ranges opened up + per-env weed color."""

    randomize_weed_color = EventTerm(
        func=mdp.randomize_weed_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("weed"),
            "base_color": WEED_COLOR,
            "variation": 0.08,
        },
    )

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        # Camera mount tolerance: per-env static offset baked at startup.
        # ~1.5 cm / ~1.7 deg (1 sigma) covers a hand-bolted RealSense mount.
        self.randomize_camera_pose.params["pos_std"] = (0.015, 0.015, 0.015)
        self.randomize_camera_pose.params["rpy_std"] = (0.03, 0.03, 0.03)
        # Lighting: wider intensity and stronger color casts.
        self.randomize_dome.params["intensity_range"] = (200.0, 700.0)
        self.randomize_dome.params["color_variation"] = 0.20
        self.randomize_sun.params["intensity_range"] = (500.0, 2200.0)
        self.randomize_sun.params["color_variation"] = 0.20
        self.randomize_mat_color.params["variation"] = 0.05


@configclass
class VolcaniarmReachVisionAmeStage2EnvCfg(VolcaniarmReachVisionAmeEnvCfg):
    events: Stage2EventCfg = Stage2EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.mask.image.params["jitter"] = GREEN_JITTER_STAGE2
        self.observations.mask.image.params["dropout"] = MASK_DROPOUT_STAGE2
        self.observations.mask.image.noise = Unoise(n_min=-0.08, n_max=0.08)


@configclass
class VolcaniarmReachVisionAmeStage2EnvCfg_PLAY(VolcaniarmReachVisionAmeStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.mask.enable_corruption = False
        # `enable_corruption` only gates the additive noise; the dropout is a
        # term param and must be cleared explicitly for clean evaluation.
        self.observations.mask.image.params["dropout"] = None
