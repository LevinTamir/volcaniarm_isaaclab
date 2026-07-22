# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Green-coverage map observation term: the AME "elevation map" analogue.

AME feeds its CNN a *structured* map (elevation xyz), not raw sensor data.
Our analogue is an HSV green mask over the camera frame: the scene is
deliberately simplified so the only green object in view is the target weed.
Everything else the arm-mounted RealSense sees — steel linkages, the red
frame, the black mat, white walls — is either unsaturated or the wrong hue,
so it all maps to ~0 and the background contributes no signal.

The math itself lives in `..mask_ops`, which imports torch and nothing else so
it can also run against real camera frames with no simulator. This module is
only the IsaacLab observation-term wrapper around it, and the ONNX bundle
calls the same functions — so sim/deploy parity is structural rather than
conventional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..mask_ops import (  # noqa: F401  -- re-exported for existing importers
    GREEN_NOMINAL,
    LCC_ITERS,
    LCC_KERNEL,
    MASK_H,
    MASK_W,
    isolate_blob,
    rgb_to_green_mask,
)
from ..contract import CAM_H, CAM_W, GREEN_JITTER

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


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
        self._dropout = cfg.params.get("dropout", None)
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

    def _apply_dropout(self, mask: torch.Tensor) -> torch.Tensor:
        """Patch + frame dropout, applied BEFORE `isolate_blob`.

        Pre-isolate placement is deliberate: real occlusion punches holes in
        the RGB-derived mask and the deployed `isolate_blob` then keeps the
        component holding the brightest surviving pixel — training must see
        that same "blob split, one fragment survives" artifact, not a
        cosmetic post-isolate hole. Frame dropout stands in for stale/black
        frames at deploy.
        """
        n, h, w = mask.shape
        dev = mask.device
        patch_prob = float(self._dropout.get("patch_prob", 0.0))
        if patch_prob > 0.0:
            n_lo, n_hi = self._dropout.get("num_patches", (1, 3))
            s_lo, s_hi = self._dropout.get("patch_size", (4, 12))
            affected = torch.rand(n, device=dev) < patch_prob
            for i in affected.nonzero(as_tuple=True)[0].tolist():
                for _ in range(int(torch.randint(n_lo, n_hi + 1, (1,)).item())):
                    ph = int(torch.randint(s_lo, s_hi + 1, (1,)).item())
                    pw = int(torch.randint(s_lo, s_hi + 1, (1,)).item())
                    r = int(torch.randint(0, max(1, h - ph), (1,)).item())
                    c = int(torch.randint(0, max(1, w - pw), (1,)).item())
                    mask[i, r : r + ph, c : c + pw] = 0.0
        frame_prob = float(self._dropout.get("frame_drop_prob", 0.0))
        if frame_prob > 0.0:
            dropped = torch.rand(n, device=dev) < frame_prob
            mask[dropped] = 0.0
        return mask

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        out_hw: tuple[int, int] | None = None,
        isolate: bool = True,
        jitter: dict | None = None,
        dropout: dict | None = None,
    ) -> torch.Tensor:
        rgb = env.scene.sensors[self._sensor_name].data.output["rgb"]
        # The camera renders at the D435i aspect (RENDER_HxRENDER_W); squash
        # to the CAM_HxCAM_W contract with the same non-uniform bilinear
        # resize the deployed controller applies to the real 848x480 stream
        # (cv::resize INTER_LINEAR). Train-time pixels == deploy-time pixels.
        if tuple(rgb.shape[1:3]) != (CAM_H, CAM_W):
            x = rgb.permute(0, 3, 1, 2).float()
            if rgb.dtype == torch.uint8:
                x = x / 255.0
            x = torch.nn.functional.interpolate(
                x, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False
            )
            rgb = x.permute(0, 2, 3, 1)
        mask = rgb_to_green_mask(
            rgb, softness=self._softness, out_hw=self._out_hw, **self._thr
        )
        if self._dropout is not None:
            mask = self._apply_dropout(mask)
        if self._isolate:
            mask = isolate_blob(mask)
        # ObservationManager does not auto-flatten; terms must be 2-D.
        return mask.flatten(1)
