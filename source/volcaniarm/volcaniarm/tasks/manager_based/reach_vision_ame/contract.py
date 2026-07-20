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

# Mask resolution fed to the CNN. We threshold at full camera res and
# average-pool down by 2, which gives anti-aliased sub-pixel coverage;
# thresholding an already-downsampled image would alias the small target.
MASK_H, MASK_W = 48, 48

# Name of the observation group carrying the flattened mask.
MASK_GROUP = "mask"

# Proprio term order inside the "policy" observation group. The env cfg
# declares its ObsTerms in this order and the ONNX bundle concatenates its
# inputs in this order. The exporter asserts the two agree.
PROPRIO_TERMS = ("joint_pos", "actions")
PROPRIO_DIM = 4

# Iterations of seeded morphological reconstruction used to isolate a single
# green blob. Propagation is (kernel-1)/2 px per step, so with a 7x7 kernel
# at 48x48 we need ~ceil(48*sqrt(2)/3) = 23 to cover the full diagonal.
LCC_ITERS = 24
LCC_KERNEL = 7

# ---------------------------------------------------------------------
# Green segmentation thresholds (normalised HSV, all in [0,1])
# ---------------------------------------------------------------------
# Starting point: the field-validated OpenCV values already deployed in
# volcaniarm_ws/src/volcaniarm_weed_detector/.../weed_detection_node.py:64-66
#   lower=[35,60,60]  upper=[85,255,255]   (OpenCV H is 0..179)
# -> hue 35..85 of 180 = 0.194..0.472, centre 0.333, halfwidth 0.139
#
# That band was calibrated for *real* foliage. The current target is a
# 3D-printed plastic weed whose green is visibly cyan-shifted — estimated
# at H~76 (0.42 normalised) from the lab photo, close to the band's upper
# edge. We therefore widen and re-centre to cover both the plastic print and
# natural green, rather than sitting on the edge.
#
# TODO(calibration): replace with the median HSV measured over weed pixels
# in a real RealSense frame. The photo estimate is a phone image with unknown
# white balance and is not authoritative. See verification step 4 in the plan.
# Calibrated against the ACTUAL rendered frame, not the material constant —
# lighting and tonemapping shift the weed's colour in the image. Measured over
# green-dominant pixels in the AME scene:
#     hue  p05=0.4222  median=0.4383  p95=0.4431   (OpenCV H ~79)
#     sat  p05=0.3152  median=0.4576  p95=0.4878
#     val  p05=0.2157  median=0.6235  p95=0.7176
# The previous values (centre 0.375, sat/val floors 0.2353) capped the mask at
# 0.81: the hue was 0.06 off-centre, and val_min sat *above* the 5th percentile
# so shadowed weed pixels were cut entirely.
#
# The floors are safe to lower because saturation, not value, is what
# discriminates against the black mat — the mat is ~(0.04,0.04,0.04), i.e.
# effectively zero saturation, so it can never enter the band however dim the
# value gate gets.
GREEN_NOMINAL = dict(
    hue_center=0.435,      # OpenCV H ~78
    hue_halfwidth=0.120,   # covers H ~57..100; still admits natural green at 60
    sat_min=0.18,
    val_min=0.12,
    softness=0.04,         # sigmoid width; ~7 OpenCV-hue units of soft edge
)

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
# 1.5x the authored STL height. Must match the target in convert_weed.py.
WEED_HEIGHT_M = 0.11
WEED_SPAWN_Z_WORLD = WEED_HEIGHT_M

# X is pinned: the 5-bar is planar and the EE only ever reaches x=0.071.
WEED_X_BASE = 0.071

# Y sampling range, measured not assumed — see scripts/check_workspace.py.
# The EE floor is z=0.052 m (the arm CANNOT reach the mat at any joint angle),
# and the reachable Y span is height-dependent. In the 0.100-0.125 m band the
# envelope is y in [-0.103, 0.143] within the reward's in-range elbow bound,
# so a symmetric +-0.10 sits inside it with margin on both sides.
#
# NOTE: the older `reach` / `reach_vision` tasks sample y in (-0.50, 0.50)
# while the reachable envelope is only [-0.287, 0.299] — roughly 40% of their
# targets were never reachable. Left as-is on purpose so their baselines still
# describe what was actually trained.
WEED_Y_RANGE = (-0.10, 0.10)

# Canopy height jitter, applied on top of WEED_SPAWN_Z_WORLD. Small and
# one-sided-upward: it stands in for weeds of slightly different size rather
# than a weed floating off the mat, and it keeps the target inside the band
# the Y range was measured for. Set to (0.0, 0.0) to pin the canopy exactly.
WEED_Z_JITTER = (0.0, 0.03)
