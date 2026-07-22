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


def _balanced_color_jitter(base: tuple[float, float, float], variation: float) -> tuple[float, float, float]:
    """Jitter `base` color while preserving brightness (zero-sum offsets)."""
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    avg = sum(offsets) / 3.0
    balanced = [o - avg for o in offsets]
    return tuple(max(0.0, min(1.0, b + o)) for b, o in zip(base, balanced))


def _find_diffuse_color_attr(prim, _depth: int = 0):
    """Walk the subtree looking for a Shader prim with a `diffuseColor` input.

    Matches both UsdPreviewSurface (input name `diffuseColor`) and the
    inputs Isaac's PreviewSurfaceCfg authors during spawn. Depth-bounded
    recursion: spawned props nest the shader at varying depth (e.g.
    <prim>/Looks/Material/Shader for a UsdFileCfg-spawned mesh).
    """
    from pxr import UsdShade

    if _depth > 6:
        return None
    if prim.IsA(UsdShade.Shader):
        attr = prim.GetAttribute("inputs:diffuseColor")
        if attr.IsValid():
            return attr
    for child in prim.GetAllChildren():
        attr = _find_diffuse_color_attr(child, _depth + 1)
        if attr is not None:
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


class reset_weed_in_reachable_workspace(ManagerTermBase):
    """Sample the weed canopy (y, z) uniformly inside the measured envelope.

    z ~ U(z_range); y ~ U(y_min(z) + margin, y_max(z) - margin), where the
    per-z y-interval comes from the generated `workspace_table` module
    (measured by scripts/check_workspace.py --emit-table). X is pinned by
    the planar mechanism. Serves both mode="reset" and mode="interval" —
    the event manager passes the due env_ids the same way for both.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        from ..workspace_table import TABLE

        z_range = cfg.params["z_range"]
        y_margin = cfg.params["y_margin"]
        rows = [r for r in TABLE if r[1] > z_range[0] and r[0] < z_range[1]]
        if not rows:
            raise ValueError(f"workspace_table has no rows covering z_range={z_range}")
        # Contiguity + coverage: the sampler lerps inside a row and must
        # never draw a z that falls between rows.
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
            # Curriculum hook: shrink each row's interval toward its centre.
            # z_range and y_scale are read per call, so the curriculum term
            # can widen them at runtime without touching the baked tensors.
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


class randomize_weed_color(ManagerTermBase):
    """Per-env diffuseColor jitter for the weed prop, resampled on reset.

    `RigidObject` exposes no `.prims`, so the per-env weed prims are resolved
    lazily by path pattern on first call and their `diffuseColor` shader
    attrs cached. Keep `variation` small: the weed must stay inside the
    (possibly widened) green HSV band or the target goes invisible to the
    actor while the critic still sees ground truth.

    If the cloner shared one material across envs (instanced spawn), every
    env resolves to the same attr — the jitter then degrades gracefully to a
    global per-reset jitter, which still decorrelates over time.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._attrs: list | None = None

    def _resolve_attrs(self, env: "ManagerBasedEnv", asset_name: str) -> list:
        import re

        import isaaclab.sim as sim_utils

        pattern = env.scene[asset_name].cfg.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
        prims = sim_utils.find_matching_prims(pattern)

        def env_index(prim) -> int:
            m = re.search(r"env_(\d+)", str(prim.GetPath()))
            return int(m.group(1)) if m else 0

        attrs = [None] * len(prims)
        for prim in prims:
            attrs[env_index(prim)] = _find_diffuse_color_attr(prim)
        return attrs

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        base_color: tuple[float, float, float],
        variation: float,
    ):
        if self._attrs is None:
            self._attrs = self._resolve_attrs(env, asset_cfg.name)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        for i in env_ids.tolist():
            attr = self._attrs[i] if i < len(self._attrs) else None
            if attr is not None:
                attr.Set(_balanced_color_jitter(base_color, variation))


class randomize_camera_pose(ManagerTermBase):
    """Give each env's camera a fixed random MOUNT offset, once, at startup.

    Fires with mode="startup": per env, samples a Gaussian offset
    (translation std `pos_std` m, rotation std `rpy_std` rad, composed in
    the camera's own frame) and bakes it into the camera prim's LOCAL pose
    on top of the spawn offset. The offset then stays constant for the
    whole run — which is the honest sim2real model: the real RealSense is
    bolted once, so its pose error is a constant extrinsic bias, not
    something that changes between episodes. Across 512 envs the policy
    still sees 512 different mount errors every rollout.

    Why local-USD writes and not `set_world_poses` per reset (the previous
    design, twice): with Fabric enabled, `XformPrimView.set_world_poses`
    writes GPU-side Fabric matrices that (a) get clobbered by the next
    hierarchy update for non-Boundable prims like cameras and (b) are never
    mirrored to USD, which is what the renderer actually consumes
    (`Camera` builds its view with sync_usd_on_fabric_write=False). Net
    effect: the old per-reset world-pose jitter NEVER moved the rendered
    image — verified empirically 2026-07-22 (a manual 5 cm write left
    `pos_w` and the render byte-identical). `set_local_poses` always goes
    through USD, which the renderer reads.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._applied = False

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        sensor_name: str,
        pos_std: tuple[float, float, float],
        rpy_std: tuple[float, float, float],
    ):
        if self._applied:
            return
        self._applied = True

        sensor = env.scene[sensor_name]
        view = sensor._view  # XformPrimView; local-pose ops use USD in all modes
        loc_t, loc_q = view.get_local_poses()
        n = loc_t.shape[0]
        dev, dtype = loc_t.device, loc_t.dtype

        pos_std_t = torch.tensor(pos_std, device=dev, dtype=dtype)
        rpy_std_t = torch.tensor(rpy_std, device=dev, dtype=dtype)

        new_t = loc_t + torch.randn(n, 3, device=dev, dtype=dtype) * pos_std_t
        rpy = torch.randn(n, 3, device=dev, dtype=dtype) * rpy_std_t
        delta_q = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        new_q = math_utils.quat_mul(loc_q, delta_q)

        view.set_local_poses(new_t, new_q)
