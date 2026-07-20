# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Green-coverage map: the AME "elevation map" analogue for this task.

AME feeds its CNN a *structured* map (elevation xyz), not raw sensor data.
Our analogue is an HSV green mask over the camera frame: the scene is
deliberately simplified so the only green object in view is the target weed.
Everything else the arm-mounted RealSense sees — steel linkages, the red
frame, the black mat, white walls — is either unsaturated or the wrong hue,
so it all maps to ~0 and the background contributes no signal.

Everything here is pure elementwise torch so the **same functions** run inside
the observation term during training and inside the exported ONNX graph at
deploy time. Sim/real parity is structural, not conventional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..contract import GREEN_JITTER, GREEN_NOMINAL, LCC_ITERS, LCC_KERNEL, MASK_H, MASK_W

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


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

    Thresholding happens at full camera resolution and only then averages
    down, which yields genuine sub-pixel coverage; thresholding an
    already-downsampled image would alias a small target.
    """
    x = rgb.permute(0, 3, 1, 2)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()

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
    # Achromatic pixels (the black mat, white walls, bare steel) have no
    # meaningful hue; pin them to 0 so they can never sit inside the band.
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
        # here — only the batch axis is dynamic, H and W are fixed at CAM_H/W.
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
    seed = mask * (mask >= peak - 1e-6).to(mask.dtype)

    seed = seed.unsqueeze(1)
    m = mask.unsqueeze(1)
    for _ in range(iters):
        seed = torch.minimum(
            torch.nn.functional.max_pool2d(seed, kernel, stride=1, padding=pad), m
        )
    return seed.squeeze(1)


class green_mask(ManagerTermBase):
    """Observation term: flattened soft green-coverage map from a TiledCamera.

    Holds per-env HSV thresholds that are resampled on reset. That jitter is
    the main sim2real hardener — without it the policy overfits to one exact
    hue window and any colour-calibration difference on the real camera breaks
    it. The ONNX export deliberately bakes the *nominal* thresholds instead;
    the jitter exists so the policy tolerates the nominal being slightly off.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._sensor_name = cfg.params["sensor_cfg"].name
        self._out_hw = tuple(cfg.params.get("out_hw", (MASK_H, MASK_W)))
        self._isolate = bool(cfg.params.get("isolate", True))
        self._jitter = dict(cfg.params.get("jitter", GREEN_JITTER))
        self._softness = GREEN_NOMINAL["softness"]
        self._thr = {
            k: torch.full((env.num_envs, 1, 1), float(v), device=env.device)
            for k, v in GREEN_NOMINAL.items()
            if k != "softness"
        }

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        for key, half in self._jitter.items():
            if key not in self._thr:
                continue
            n = self._thr[key][env_ids].shape[0]
            base = GREEN_NOMINAL[key]
            draw = (torch.rand(n, 1, 1, device=self._thr[key].device) * 2.0 - 1.0) * half
            self._thr[key][env_ids] = base + draw

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        out_hw: tuple[int, int] | None = None,
        isolate: bool = True,
        jitter: dict | None = None,
    ) -> torch.Tensor:
        rgb = env.scene.sensors[self._sensor_name].data.output["rgb"]
        mask = rgb_to_green_mask(
            rgb, softness=self._softness, out_hw=self._out_hw, **self._thr
        )
        if self._isolate:
            mask = isolate_blob(mask)
        # ObservationManager does not auto-flatten; terms must be 2-D.
        return mask.flatten(1)
