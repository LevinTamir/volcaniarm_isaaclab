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
    # 0.10, not the once-planned 0.12: these are HALF-RANGES around
    # sat_min=0.18 / val_min=0.12, and floors of 0.06/0.00 let mildly-lit
    # mat pixels through — combined with any green-ish cast that floods
    # whole frames (measured on rendered frames 2026-07-23).
    sat_min=0.100,
    val_min=0.100,
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
    """Stage-1 events with the DR ranges opened up + per-env weed color,
    weed size and arm color."""

    randomize_weed_color = EventTerm(
        func=mdp.randomize_weed_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("weed"),
            "base_color": WEED_COLOR,
            "variation": 0.08,
        },
    )

    # Per-env static weed size (real printed props come in several sizes —
    # a constant per-env bias, like the camera mount offset).
    randomize_weed_scale = EventTerm(
        func=mdp.randomize_weed_scale,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("weed"),
            "scale_range": (0.5, 1.5),
        },
    )

    # Slight per-env arm color drift. Capped at 0.05 so the red links can
    # never wander into the green HSV band and spoof the mask.
    randomize_robot_color = EventTerm(
        func=mdp.randomize_robot_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "variation": 0.05,
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
        # Lighting: wider intensity range; color casts are SMALL and on the
        # color-temperature axis only. Rationale, measured on rendered
        # frames 2026-07-23: the mask band (hue 0.315-0.555) includes
        # teal/cyan, and any green/cool cast on the near-black mat floods
        # the whole frame for the episode (balanced 0.20 flooded most
        # episodes; balanced 0.10 ~1/2; temp 0.15 still flooded because the
        # sun's base already has green at max, making cool casts cyan).
        # The real camera auto-white-balances global casts away anyway, so
        # big casts mismodel deploy; residual calibration error is carried
        # by GREEN_JITTER_STAGE2, and the heavy lifting by intensity,
        # shadow-direction and mask-dropout DR.
        # temp_variation 0.10 (was 0.06): real D435i frames (2026-07-23)
        # show the daylight cast leaves the mat at saturation up to ~0.23
        # even after auto-white-balance — 0.06 under-modelled it. Gated by
        # the flood check; walk back to 0.08 first if it floods.
        self.randomize_dome.params["intensity_range"] = (200.0, 700.0)
        self.randomize_dome.params["color_variation"] = 0.0
        self.randomize_dome.params["temp_variation"] = 0.10
        self.randomize_sun.params["intensity_range"] = (500.0, 2200.0)
        self.randomize_sun.params["color_variation"] = 0.0
        self.randomize_sun.params["temp_variation"] = 0.10
        # Shadow direction/length (sun re-aimed per reset) and softness.
        self.randomize_sun.params["elevation_range_deg"] = (30.0, 80.0)
        self.randomize_sun.params["azimuth_range_deg"] = (0.0, 360.0)
        self.randomize_sun.params["angle_range"] = (1.0, 10.0)
        # Mat: still black, but different blacks — vary brightness and
        # matte-to-semigloss, NOT chroma. Even variation=0.02 is huge
        # RELATIVE to a 0.02-0.08 grey (up to ~0.4 saturation on a dark
        # mat) and was the last remaining mask-flood source in the
        # 2026-07-23 frame measurements. The physical rubber mat is
        # achromatic; scene hue variation comes from the light casts.
        # brightness top 0.15 (was 0.08): the real camera's auto-exposure
        # lifts the black mat to rendered V 0.27-0.54 (measured 2026-07-23)
        # — brighter than the sim ever showed it. Brightness alone is
        # flood-safe (floods need chroma); gated by the flood check.
        self.randomize_mat_color.params["variation"] = 0.005
        self.randomize_mat_color.params["brightness_range"] = (0.02, 0.15)
        self.randomize_mat_color.params["roughness_range"] = (0.4, 1.0)


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
