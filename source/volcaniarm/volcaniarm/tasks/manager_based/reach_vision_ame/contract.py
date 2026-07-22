# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Frozen interface shared by the env cfg, the policy module, and the ONNX exporter.

Every consumer imports from here. Nothing downstream hardcodes these numbers.

The reason this module exists: the old vision task carried an *implicit*
observation-ordering contract duplicated across the env cfg, the exporter's
`torch.cat`, and the C++ controller's input packing — guarded only by a
comment. If those drifted, the arm moved wrongly with no error raised
anywhere. One definition kills that whole failure mode.
"""

# Camera resolution. Must match `image_width`/`image_height` in
# volcaniarm_ws/.../config/volcaniarm_rl_vision_controller.yaml — the C++
# controller resizes incoming frames to exactly this before inference.
CAM_H, CAM_W = 96, 96

# Training-camera RENDER resolution — the RealSense D435i color aspect
# (848x480 -> 96x170 at CAM_H rows). The real controller squashes 848x480
# down to CAM_HxCAM_W (cv::resize, non-uniform); training renders at the
# same aspect and squashes identically in the green_mask obs term, so the
# policy sees the same anamorphic geometry in sim as on the robot. The
# camera FOV (69 x ~42.5 deg, set in the env cfg) matches the D435i color
# sensor for the same reason.
RENDER_H, RENDER_W = 96, 170

# Deployment-facing mask math and its thresholds live in `mask_ops`, which
# imports torch and nothing else so it can run against real camera frames
# without launching Isaac. Re-exported here so consumers keep treating this
# module as the single source of truth.
from .mask_ops import (  # noqa: F401,E402
    GREEN_NOMINAL,
    LCC_ITERS,
    LCC_KERNEL,
    MASK_H,
    MASK_W,
    isolate_blob,
    rgb_to_green_mask,
)

# Name of the observation group carrying the flattened mask.
MASK_GROUP = "mask"

# Proprio term order inside the "policy" observation group. The env cfg
# declares its ObsTerms in this order and the ONNX bundle concatenates its
# inputs in this order. The exporter asserts the two agree.
PROPRIO_TERMS = ("joint_pos", "actions")
PROPRIO_DIM = 4

# Per-env jitter half-ranges, resampled on reset. Sim only — the export bakes
# GREEN_NOMINAL. This is what makes the policy robust to the nominal being
# slightly wrong on the real robot, which is the dominant sim2real risk here.
GREEN_JITTER = dict(
    hue_center=0.030,
    hue_halfwidth=0.030,
    sat_min=0.080,
    val_min=0.080,
)

# Sim material for the printed weed, bound at spawn via UsdFileCfg. Estimated
# from the lab photo to sit at hue ~0.42; re-measure alongside GREEN_NOMINAL.
WEED_COLOR = (0.24, 0.75, 0.51)

# Matte black rubber mat. Near-zero saturation, so it never trips the green
# mask regardless of exact value; high roughness keeps it non-reflective.
MAT_COLOR = (0.04, 0.04, 0.04)
MAT_ROUGHNESS = 0.9

# ---------------------------------------------------------------------
# Weed placement
# ---------------------------------------------------------------------
# Built by scripts/convert_weed.py with its ORIGIN AT THE APEX, so the prim's
# root pose is the canopy — i.e. the reach target — and the mesh hangs down to
# the mat. Spawning at world z = WEED_HEIGHT_M therefore seats the base at
# z=0 while `position_weed_error` / `weed_pos_in_base` keep working against
# root_pos_w with no offset term to maintain.
# ~1.57x the authored STL height. Must match the target in convert_weed.py
# and WEED_HEIGHT in scripts/build_lab.py.
WEED_HEIGHT_M = 0.115
WEED_SPAWN_Z_WORLD = WEED_HEIGHT_M

# X is pinned: the 5-bar is planar and the EE only ever reaches x=0.071.
WEED_X_BASE = 0.071

# Y sampling range, measured not assumed — see scripts/check_workspace.py.
# The EE floor is z=0.052 m (the arm CANNOT reach the mat at any joint angle),
# and the reachable Y span is height-dependent. In the 0.100-0.125 m band the
# envelope is y in [-0.103, +0.143] within the reward's in-range elbow bound.
#
# ASYMMETRIC ON PURPOSE. The 5-bar reaches ~4 cm further in +Y than in -Y at
# every height, so a symmetric +-0.10 put the negative extreme within 3 mm of
# the boundary while leaving 4 cm of slack on the positive side — targets at
# y=-0.10 were effectively at the edge of feasibility, which is what "the arm
# struggles at the outer sides" looked like. These bounds keep ~2 cm of margin
# at both ends.
#
# NOTE: the older `reach` / `reach_vision` tasks sample y in (-0.50, 0.50)
# while the reachable envelope is only [-0.287, 0.299] — roughly 40% of their
# targets were never reachable. Left as-is on purpose so their baselines still
# describe what was actually trained.
WEED_Y_RANGE = (-0.08, 0.12)

# Canopy height jitter, applied on top of WEED_SPAWN_Z_WORLD. Small and
# one-sided-upward: it stands in for weeds of slightly different size rather
# than a weed floating off the mat.
#
# Kept tight (canopy 0.115-0.130) because the reachable Y span *shrinks* with
# height once the elbows approach their bound — at 0.03 the canopy could land
# in the 0.125-0.150 band, where Y narrows enough that the outer targets stop
# being reachable at all. Set to (0.0, 0.0) to pin the canopy exactly.
WEED_Z_JITTER = (0.0, 0.015)
