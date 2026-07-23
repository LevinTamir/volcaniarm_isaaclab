# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Green-mask math and its thresholds. Imports **torch and nothing else**.

This module is deliberately free of IsaacLab imports, and uses no relative
imports, so it can be loaded in a plain Python session with no simulator:

    import importlib.util
    spec = importlib.util.spec_from_file_location("mask_ops", "<path>/mask_ops.py")
    mask_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_ops)

That matters because the most valuable pre-deployment check runs these exact
functions over **real RealSense frames** and compares them against the OpenCV
detector in `volcaniarm_ws/src/volcaniarm_weed_detector/` — a test that needs
no Isaac Sim and should not require a 2-minute app launch. Importing anything
from `volcaniarm.tasks` pulls in the whole IsaacLab stack down to `pxr`, so
the deployment-facing math lives here instead.

`contract.py` re-exports these names, so in-package consumers can keep
importing from the contract as the single source of truth.
"""

from __future__ import annotations

import torch

# Mask resolution fed to the CNN. We threshold at full camera resolution and
# average-pool down by 2, which gives anti-aliased sub-pixel coverage;
# thresholding an already-downsampled image would alias the small target.
MASK_H, MASK_W = 48, 48

# Iterations of seeded morphological reconstruction used to isolate a single
# green blob. Propagation is (kernel-1)/2 px per step, so with a 7x7 kernel at
# 48x48 we need ~ceil(48*sqrt(2)/3) = 23 to cover the full diagonal.
LCC_ITERS = 24
LCC_KERNEL = 7

# ---------------------------------------------------------------------
# Green segmentation thresholds (normalised HSV, all in [0,1])
# ---------------------------------------------------------------------
# Lineage: started from the field-validated OpenCV values deployed in
# volcaniarm_ws/src/volcaniarm_weed_detector/.../weed_detection_node.py:64-66
#   lower=[35,60,60]  upper=[85,255,255]   (OpenCV H is 0..179)
# then re-centred against the ACTUAL rendered frame, because lighting and
# tonemapping shift the weed's colour in the image away from its material
# constant. Measured over green-dominant pixels in the AME scene:
#     hue  p05=0.4222  median=0.4383  p95=0.4431   (OpenCV H ~79)
#     sat  p05=0.3152  median=0.4576  p95=0.4878
#     val  p05=0.2157  median=0.6235  p95=0.7176
# The earlier values (centre 0.375, sat/val floors 0.2353) capped the mask at
# 0.81: hue was 0.06 off-centre and val_min sat *above* the 5th percentile, so
# shadowed weed pixels were cut entirely.
#
# The floors are safe this low because *saturation*, not value, discriminates
# against the black mat — the mat is ~(0.04,0.04,0.04), effectively zero
# saturation, so it cannot enter the band however low the value gate goes.
#
# CALIBRATED against real D435i frames (2026-07-23, 5 frames via
# grab_camera_frames.py + a hue_center x hue_halfwidth sweep with deploy
# metrics: post-isolate blob strength/size, peak-on-weed, background
# leakage). The real printed weed renders with hue spread 0.26-0.50 (dark
# body ~0.374, bright saturated highlights ~0.50); the bright pixels
# dominate the blob, and THIS band measured optimal or equal-best on every
# metric (peak 0.82, clean single blob, zero mat response — the mat's
# blue-daylight cast at hue 0.49-0.54 falls inside the band but is
# rejected by the saturation gate: mat S<=0.23 vs weed S~0.31-0.51).
# Recentering toward the body median only weakened the blob. KEEP AS IS.
GREEN_NOMINAL = dict(
    hue_center=0.435,      # OpenCV H ~78
    hue_halfwidth=0.120,   # covers H ~57..100; still admits natural green at 60
    sat_min=0.18,
    val_min=0.12,
    softness=0.04,         # sigmoid width; ~7 OpenCV-hue units of soft edge
)


def rgb_to_green_mask(
    rgb: torch.Tensor,
    hue_center: float | torch.Tensor = GREEN_NOMINAL["hue_center"],
    hue_halfwidth: float | torch.Tensor = GREEN_NOMINAL["hue_halfwidth"],
    sat_min: float | torch.Tensor = GREEN_NOMINAL["sat_min"],
    val_min: float | torch.Tensor = GREEN_NOMINAL["val_min"],
    softness: float = GREEN_NOMINAL["softness"],
    out_hw: tuple[int, int] | None = (MASK_H, MASK_W),
) -> torch.Tensor:
    """RGB -> soft green-coverage map in [0, 1].

    Args:
        rgb: ``(N, H, W, 3)``, uint8 in [0,255] or float in [0,1].
        hue_center/hue_halfwidth/sat_min/val_min: normalised HSV thresholds.
            Scalars on the export path; ``(N,1,1)`` tensors for per-env
            randomisation in sim. Both broadcast identically.
        softness: sigmoid width. Larger = softer threshold edge.
        out_hw: average-pool the mask down to this size, or None to skip.

    Returns:
        ``(N, out_h, out_w)`` float32.

    The mask is **soft** (a product of three sigmoids) rather than binary.
    That is not about gradients — nothing backprops through an observation —
    but because a few degrees of hue mismatch between sim and the real
    RealSense then degrades the value smoothly instead of dropping the blob
    to zero, which is the dominant sim2real failure mode here.
    """
    x = rgb.permute(0, 3, 1, 2)
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()

    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx = x.max(dim=1).values
    mn = x.min(dim=1).values
    d = mx - mn
    eps = 1e-6

    # Hue in [0,1). Branch explicitly rather than using a float modulo — the
    # `where` form has no wraparound edge case and exports more predictably.
    h_r = torch.where(g >= b, (g - b) / (d + eps), (g - b) / (d + eps) + 6.0)
    h_g = (b - r) / (d + eps) + 2.0
    h_b = (r - g) / (d + eps) + 4.0
    h = torch.where(mx == r, h_r, torch.where(mx == g, h_g, h_b)) / 6.0
    # Achromatic pixels (black mat, white walls, bare steel) have no meaningful
    # hue; pin them to 0 so they can never sit inside the band.
    h = torch.where(d < eps, torch.zeros_like(h), h)

    s = d / (mx + eps)
    v = mx

    dh = torch.abs(h - hue_center)
    dh = torch.minimum(dh, 1.0 - dh)  # hue is circular

    mask = (
        torch.sigmoid((hue_halfwidth - dh) / softness)
        * torch.sigmoid((s - sat_min) / softness)
        * torch.sigmoid((v - val_min) / softness)
    )

    if out_hw is not None and (int(mask.shape[-2]), int(mask.shape[-1])) != tuple(out_hw):
        # int() is load-bearing under torch.onnx.export: tracing turns
        # `mask.shape[-2]` into a Tensor, and avg_pool2d rejects a tensor
        # kernel_size ("must either be a single int, or a tuple of two ints").
        # Forcing a Python int bakes the factor as a constant, which is correct
        # here — only the batch axis is dynamic, H and W are fixed.
        k = int(mask.shape[-2]) // int(out_hw[0])
        mask = torch.nn.functional.avg_pool2d(mask.unsqueeze(1), k).squeeze(1)
    return mask


def isolate_blob(
    mask: torch.Tensor,
    iters: int = LCC_ITERS,
    kernel: int = LCC_KERNEL,
) -> torch.Tensor:
    """Keep only the connected component containing the mask's brightest pixel.

    True connected-component labelling is not ONNX-exportable — it needs
    data-dependent iterative labelling. This is the exportable equivalent:
    grayscale **morphological reconstruction by dilation**, seeded at the peak
    and clamped by the mask, unrolled a fixed number of times.

    Note the semantics: this returns the component containing the *brightest*
    pixel, not the one with the largest *area*. For a single weed the two
    coincide, and peak-seeding is arguably more robust — a large dim false
    region loses to a small bright true one. Genuine largest-by-area needs
    per-component area sums, which is precisely the part that will not export.

    Propagation is ``(kernel-1)/2`` px per iteration, so `iters` must cover the
    mask diagonal or a large blob will be clipped part-way.
    """
    pad = kernel // 2
    peak = mask.amax(dim=(-2, -1), keepdim=True)
    # Seed = the peak pixels only. Multiplying by `mask` keeps it grayscale so
    # reconstruction stays soft rather than snapping to binary.
    seed = (mask * (mask >= peak - 1e-6).to(mask.dtype)).unsqueeze(1)
    m = mask.unsqueeze(1)
    for _ in range(iters):
        seed = torch.minimum(
            torch.nn.functional.max_pool2d(seed, kernel, stride=1, padding=pad), m
        )
    return seed.squeeze(1)
