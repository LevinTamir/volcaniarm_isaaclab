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

# ---------------------------------------------------------------------
# (y, z) sampling — table-driven since 2026-07-22
# ---------------------------------------------------------------------
# The canopy is sampled anywhere in the measured reachable envelope: z
# uniform in WEED_Z_RANGE_WORLD, then y uniform in the per-z interval from
# the generated `workspace_table.py` (reachable AND camera-visible), shrunk
# by WEED_Y_SAFETY_MARGIN per side. Above ~0.13 the weed floats over the
# mat — it stands in for taller weeds; the mask pipeline does not care.
#
# The envelope was RE-MEASURED 2026-07-22 with a corrected sweep
# (quasi-static ramp from home + working-branch filter; the old teleport
# sweep let the drives pull toward home during settling, so its numbers —
# including the old "EE floor z=0.052" and "y in [-0.103, 0.143]" — were
# wrong). At the ±75° elbow bound the true canopy-band envelope is about
# y in [-0.35, +0.47] at z=0.115, widening with height. Widening the elbow
# bound past ±75° gains nothing below z=0.30, so the reward bounds stay.
WEED_Z_RANGE_WORLD = (0.115, 0.25)
WEED_Y_SAFETY_MARGIN = 0.02

# SUPERSEDED by the table sampling above (kept for reference: this was the
# line the pre-2026-07-22 checkpoints trained on).
WEED_Y_RANGE = (-0.08, 0.12)
WEED_Z_JITTER = (0.0, 0.015)
