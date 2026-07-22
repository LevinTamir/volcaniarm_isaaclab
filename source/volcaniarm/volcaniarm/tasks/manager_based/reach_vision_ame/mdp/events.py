# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Domain-randomization helpers for the vision reach task.

These functions are wired up as `EventTerm`s in `reach_vision_env_cfg.py`.
All helpers fire on reset of the affected envs unless noted otherwise.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import AssetBase
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class reset_weed_in_reachable_workspace(ManagerTermBase):
    """Sample the weed canopy (y, z) uniformly inside the measured envelope.

    z ~ U(z_range); y ~ U(y_min(z) + margin, y_max(z) - margin), where the
    per-z y-interval comes from the generated `workspace_table` module
    (measured by scripts/check_workspace.py --emit-table under the
    joystick-derived mirrored joint bounds). X is pinned by the planar
    mechanism. Serves both mode="reset" and mode="interval" — the event
    manager passes the due env_ids the same way for both.

    `y_scale` (default 1.0, full region) shrinks each row's interval toward
    its centre — a dormant hook for a spatial curriculum; nothing drives it
    in the current single-stage design.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        from ..workspace_table import TABLE

        z_range = cfg.params["z_range"]
        y_margin = cfg.params["y_margin"]
        rows = [r for r in TABLE if r[1] > z_range[0] and r[0] < z_range[1]]
        if not rows:
            raise ValueError(f"workspace_table has no rows covering z_range={z_range}")
        # Contiguity + coverage: the sampler picks a row by z and must never
        # draw a z that falls between rows.
        for a, b in zip(rows[:-1], rows[1:]):
            if abs(a[1] - b[0]) > 1e-6:
                raise ValueError(f"workspace_table gap between z={a[1]} and z={b[0]}")
        if rows[0][0] > z_range[0] + 1e-6 or rows[-1][1] < z_range[1] - 1e-6:
            raise ValueError(
                f"workspace_table rows cover [{rows[0][0]}, {rows[-1][1]}] "
                f"but z_range={z_range} — regenerate the table or shrink the band"
            )
        too_narrow = [r for r in rows if (r[3] - r[2]) <= 2.0 * y_margin]
        if too_narrow:
            raise ValueError(
                f"{len(too_narrow)} table rows narrower than 2*y_margin={2*y_margin}: "
                f"first={too_narrow[0]} — shrink z_range or y_margin"
            )
        dev = env.device
        self._z_lo = torch.tensor([r[0] for r in rows], device=dev)
        self._z_hi = torch.tensor([r[1] for r in rows], device=dev)
        self._y_min = torch.tensor([r[2] + y_margin for r in rows], device=dev)
        self._y_max = torch.tensor([r[3] - y_margin for r in rows], device=dev)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        x_pos: float,
        z_range: tuple[float, float],
        y_margin: float,
        y_scale: float = 1.0,
    ):
        asset = env.scene[asset_cfg.name]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        n = len(env_ids)
        dev = env.device

        z = torch.empty(n, device=dev).uniform_(*z_range)
        idx = torch.clamp(
            torch.searchsorted(self._z_hi, z, right=True), max=len(self._z_hi) - 1
        )
        y_lo, y_hi = self._y_min[idx], self._y_max[idx]
        if y_scale < 1.0:
            c = 0.5 * (y_lo + y_hi)
            y_lo = c - (c - y_lo) * y_scale
            y_hi = c + (y_hi - c) * y_scale
        y = y_lo + torch.rand(n, device=dev) * (y_hi - y_lo)

        pos = env.scene.env_origins[env_ids] + torch.stack(
            [torch.full((n,), x_pos, device=dev), y, z], dim=-1
        )
        quat = asset.data.default_root_state[env_ids, 3:7]
        asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=dev), env_ids=env_ids
        )


def _balanced_color_jitter(base: tuple[float, float, float], variation: float) -> tuple[float, float, float]:
    """Jitter `base` color while preserving brightness (zero-sum offsets)."""
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    avg = sum(offsets) / 3.0
    balanced = [o - avg for o in offsets]
    return tuple(max(0.0, min(1.0, b + o)) for b, o in zip(base, balanced))


def _find_diffuse_color_attr(prim):
    """Walk children looking for a Shader prim with a `diffuseColor` input.

    Matches both UsdPreviewSurface (input name `diffuseColor`) and the
    inputs Isaac's PreviewSurfaceCfg authors during spawn.
    """
    from pxr import UsdShade

    for child in prim.GetAllChildren():
        if child.IsA(UsdShade.Shader):
            attr = child.GetAttribute("inputs:diffuseColor")
            if attr.IsValid():
                return attr
        # Recurse one level — PreviewSurfaceCfg sometimes wraps shaders
        # in an intermediate scope (e.g. <prim>/Looks/Material/Shader).
        for grandchild in child.GetAllChildren():
            if grandchild.IsA(UsdShade.Shader):
                attr = grandchild.GetAttribute("inputs:diffuseColor")
                if attr.IsValid():
                    return attr
    return None


def randomize_visual_color_global(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    base_color: tuple[float, float, float],
    variation: float,
):
    """Jitter the diffuseColor of a global visual asset's bound material.

    Targets a single-prim global asset (e.g. the floor cuboid). With
    `replicate_physics=True` the asset is one prim regardless of env count,
    so the new color applies to every env's view from the next render
    onward. The function does nothing per-env-batch; it just samples once
    and writes the attribute.
    """
    asset: AssetBase = env.scene[asset_cfg.name]
    prim = asset.prims[0]
    color_attr = _find_diffuse_color_attr(prim)
    if color_attr is None:
        return
    color = _balanced_color_jitter(base_color, variation)
    color_attr.Set(color)


def randomize_light(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    intensity_range: tuple[float, float],
    color_base: tuple[float, float, float] = (1.0, 1.0, 1.0),
    color_variation: float = 0.0,
):
    """Set a light's `inputs:intensity` (and optionally `inputs:color`).

    Lights are global scene assets — one prim, one set per call. Useful for
    `DomeLightCfg` / `DistantLightCfg`.
    """
    asset: AssetBase = env.scene[asset_cfg.name]
    prim = asset.prims[0]

    intensity_attr = prim.GetAttribute("inputs:intensity")
    if intensity_attr.IsValid():
        intensity_attr.Set(random.uniform(*intensity_range))

    if color_variation > 0.0:
        color_attr = prim.GetAttribute("inputs:color")
        if color_attr.IsValid():
            color = _balanced_color_jitter(color_base, color_variation)
            color_attr.Set(color)


def randomize_camera_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    sensor_name: str,
    pos_std: tuple[float, float, float],
    rpy_std: tuple[float, float, float],
):
    """Jitter the camera's world pose per env, additively on each reset.

    Reads the affected envs' current camera world pose, adds small Gaussian
    noise on translation (per-axis std `pos_std`, m) and rotation (per-axis
    std `rpy_std`, rad) about the current frame, and writes back via
    `set_world_poses(convention="world")`.

    The jitter accumulates across resets — over many resets the camera
    drifts. For 30 Hz × 12 s episodes that's ~12k resets per env over a
    1k-iter run; pos_std=1e-3 gives expected drift ~σ·√N ≈ 11 cm at the
    end. Keep `pos_std` and `rpy_std` small. If drift becomes a problem,
    track a per-env base pose at startup and recompose against the parent
    body's world pose every reset (TODO).
    """
    sensor = env.scene[sensor_name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    pos_w = sensor.data.pos_w[env_ids]
    quat_w = sensor.data.quat_w_world[env_ids]

    pos_std_t = torch.tensor(pos_std, device=env.device, dtype=pos_w.dtype)
    rpy_std_t = torch.tensor(rpy_std, device=env.device, dtype=pos_w.dtype)

    pos_noise = torch.randn_like(pos_w) * pos_std_t
    new_pos = pos_w + pos_noise

    rpy_noise = torch.randn(len(env_ids), 3, device=env.device, dtype=pos_w.dtype) * rpy_std_t
    delta_quat = math_utils.quat_from_euler_xyz(
        rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2]
    )
    new_quat = math_utils.quat_mul(quat_w, delta_quat)

    sensor.set_world_poses(new_pos, new_quat, env_ids=env_ids, convention="world")
