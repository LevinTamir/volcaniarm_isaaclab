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


class randomize_camera_pose(ManagerTermBase):
    """Jitter the camera's world pose per env about its HOME pose on each reset.

    On the first call per env the current world pose is cached as the home
    pose — at that point no jitter has ever been applied, and the camera's
    parent chain (`Robot/camera_link`) is a fixed mount on a fixed-base
    robot, so the home world pose is constant per env. Every reset then
    writes `home ⊕ fresh noise` (translation: per-axis Gaussian std
    `pos_std`, m; rotation: per-axis Gaussian std `rpy_std`, rad, composed
    about the home frame) via `set_world_poses(convention="world")`.

    Composing against home instead of the current pose is what keeps the
    jitter from accumulating: the previous function form added noise to the
    already-noised pose, which random-walked ~σ·√N over N resets.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._home_pos: torch.Tensor | None = None
        self._home_quat: torch.Tensor | None = None
        self._captured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        sensor_name: str,
        pos_std: tuple[float, float, float],
        rpy_std: tuple[float, float, float],
    ):
        sensor = env.scene[sensor_name]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        if self._home_pos is None:
            self._home_pos = torch.zeros(env.num_envs, 3, device=env.device)
            self._home_quat = torch.zeros(env.num_envs, 4, device=env.device)

        fresh = env_ids[~self._captured[env_ids]]
        if len(fresh) > 0:
            self._home_pos[fresh] = sensor.data.pos_w[fresh]
            self._home_quat[fresh] = sensor.data.quat_w_world[fresh]
            self._captured[fresh] = True

        home_pos = self._home_pos[env_ids]
        home_quat = self._home_quat[env_ids]

        pos_std_t = torch.tensor(pos_std, device=env.device, dtype=home_pos.dtype)
        rpy_std_t = torch.tensor(rpy_std, device=env.device, dtype=home_pos.dtype)

        new_pos = home_pos + torch.randn_like(home_pos) * pos_std_t

        rpy_noise = torch.randn(len(env_ids), 3, device=env.device, dtype=home_pos.dtype) * rpy_std_t
        delta_quat = math_utils.quat_from_euler_xyz(
            rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2]
        )
        new_quat = math_utils.quat_mul(home_quat, delta_quat)

        sensor.set_world_poses(new_pos, new_quat, env_ids=env_ids, convention="world")
