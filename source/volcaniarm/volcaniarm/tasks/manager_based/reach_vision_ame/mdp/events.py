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


def _find_shader_attr(prim, input_name: str, _depth: int = 0):
    """Walk the subtree looking for a Shader prim with an `inputs:<name>` attr.

    Matches the inputs Isaac's PreviewSurfaceCfg authors during spawn
    (`diffuseColor`, `roughness`, `metallic`, ...). Depth-bounded recursion:
    spawned props nest the shader at varying depth (e.g.
    <prim>/Looks/Material/Shader for a UsdFileCfg-spawned mesh).
    """
    from pxr import UsdShade

    if _depth > 6:
        return None
    if prim.IsA(UsdShade.Shader):
        attr = prim.GetAttribute(f"inputs:{input_name}")
        if attr.IsValid():
            return attr
    for child in prim.GetAllChildren():
        attr = _find_shader_attr(child, input_name, _depth + 1)
        if attr is not None:
            return attr
    return None


def _find_diffuse_color_attr(prim):
    return _find_shader_attr(prim, "diffuseColor")


def randomize_visual_color_global(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    base_color: tuple[float, float, float],
    variation: float,
    brightness_range: tuple[float, float] | None = None,
    roughness_range: tuple[float, float] | None = None,
    metallic_range: tuple[float, float] | None = None,
):
    """Jitter a global visual asset's bound material (color, optionally PBR).

    Targets a single-prim global asset (e.g. the floor cuboid). With
    `replicate_physics=True` the asset is one prim regardless of env count,
    so the new look applies to every env's view from the next render
    onward. The function does nothing per-env-batch; it just samples once
    and writes the attributes.

    Stage-2 DR extras (defaults None -> stage-1 behavior unchanged):
    - `brightness_range`: resample the base color's mean level (hue kept) —
      "different blacks" for the mat. Chroma stays governed by `variation`,
      which must stay SMALL for the mat: a chroma jitter on a near-black
      base creates saturated dark colors, and a greenish dark mat under
      bright light lands inside the mask's HSV band and floods the whole
      frame for the episode (verified on rendered frames 2026-07-23).
    - `roughness_range` / `metallic_range`: jitter the same-named
      PreviewSurface inputs — matte rubber to semi-gloss.
    """
    asset: AssetBase = env.scene[asset_cfg.name]
    prim = asset.prims[0]
    color_attr = _find_diffuse_color_attr(prim)
    if color_attr is None:
        return
    base = base_color
    if brightness_range is not None:
        mean = max(sum(base) / 3.0, 1e-6)
        scale = random.uniform(*brightness_range) / mean
        base = tuple(min(1.0, c * scale) for c in base)
    color = _balanced_color_jitter(base, variation)
    color_attr.Set(color)
    if roughness_range is not None:
        attr = _find_shader_attr(prim, "roughness")
        if attr is not None:
            attr.Set(random.uniform(*roughness_range))
    if metallic_range is not None:
        attr = _find_shader_attr(prim, "metallic")
        if attr is not None:
            attr.Set(random.uniform(*metallic_range))


def randomize_light(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    intensity_range: tuple[float, float],
    color_base: tuple[float, float, float] = (1.0, 1.0, 1.0),
    color_variation: float = 0.0,
    temp_variation: float = 0.0,
    elevation_range_deg: tuple[float, float] | None = None,
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0),
    angle_range: tuple[float, float] | None = None,
):
    """Set a light's `inputs:intensity` (and optionally color / direction).

    Lights are global scene assets — one prim, one set per call. Useful for
    `DomeLightCfg` / `DistantLightCfg`.

    Color jitter comes in two flavours:
    - `color_variation`: balanced (brightness-preserving) RGB jitter. Free
      hue — CAN land on a green/cyan cast, which tints the achromatic mat
      into the mask's HSV band and floods the whole frame for the episode
      (verified on rendered frames 2026-07-23). Keep small.
    - `temp_variation`: warm<->cool jitter along the color-temperature axis
      only (r+t, g, b-t). This is what real lighting actually does, and by
      construction its casts sit at hue 0.0/0.66 — outside the green band —
      so it can be pushed harder without flooding. Takes precedence.

    Stage-2 DR extras (default off):
    - `elevation_range_deg` + `azimuth_range_deg`: re-aim a DistantLight so
      shadow direction/length vary. Elevation is measured up from the
      horizon (90 = straight down). UsdLux lights emit along local -Z; the
      sampled direction is written to the prim's `xformOp:orient`.
    - `angle_range`: DistantLight `inputs:angle` (deg) — apparent source
      size, i.e. shadow softness.
    """
    asset: AssetBase = env.scene[asset_cfg.name]
    prim = asset.prims[0]

    intensity_attr = prim.GetAttribute("inputs:intensity")
    if intensity_attr.IsValid():
        intensity_attr.Set(random.uniform(*intensity_range))

    if temp_variation > 0.0 or color_variation > 0.0:
        color_attr = prim.GetAttribute("inputs:color")
        if color_attr.IsValid():
            if temp_variation > 0.0:
                t = random.uniform(-temp_variation, temp_variation)
                color = (
                    max(0.0, min(1.0, color_base[0] + t)),
                    color_base[1],
                    max(0.0, min(1.0, color_base[2] - t)),
                )
            else:
                color = _balanced_color_jitter(color_base, color_variation)
            color_attr.Set(color)

    if angle_range is not None:
        angle_attr = prim.GetAttribute("inputs:angle")
        if angle_attr.IsValid():
            angle_attr.Set(random.uniform(*angle_range))

    if elevation_range_deg is not None:
        import math

        from pxr import Gf, UsdGeom

        el = math.radians(random.uniform(*elevation_range_deg))
        az = math.radians(random.uniform(*azimuth_range_deg))
        # Light travel direction (world): down at `el` above the horizon.
        d = Gf.Vec3d(
            math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)
        )
        rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), d)
        xf = UsdGeom.Xformable(prim)
        orient_op = next(
            (o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeOrient),
            None,
        )
        if orient_op is None:
            orient_op = xf.AddOrientOp()
        q = rot.GetQuat()
        if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            orient_op.Set(Gf.Quatf(q))
        else:
            orient_op.Set(q)


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


class randomize_weed_scale(ManagerTermBase):
    """Per-env static uniform scale on the weed prim, baked once at startup.

    Same sim2real logic as the camera mount offset: each printed weed has
    one fixed size, so size is a constant per-env bias, not per-episode
    noise — and the real props come in several sizes. The weed USD origin
    is the canopy point the reach reward tracks, so scaling about the prim
    origin leaves the target position exact; only the visual footprint
    (mask blob size, canopy-to-mat extent) changes. Physics never reads the
    scale: collision and gravity are disabled on the weed.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._applied = False

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        scale_range: tuple[float, float] = (0.5, 1.5),
    ):
        if self._applied:
            return
        self._applied = True

        from pxr import Gf, UsdGeom

        import isaaclab.sim as sim_utils

        def scale_op_of(prim):
            xf = UsdGeom.Xformable(prim)
            return next(
                (o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeScale),
                None,
            )

        pattern = env.scene[asset_cfg.name].cfg.prim_path.format(
            ENV_REGEX_NS="/World/envs/env_.*"
        )
        prims = sim_utils.find_matching_prims(pattern)
        # Capture the pre-existing scale ONCE, before any write. The cloner
        # makes env_1..N reference env_0's prim, so an op authored on env_0
        # becomes every clone's inherited value — a per-prim read-multiply-
        # write loop would compound the source draw into the clones
        # (verified: clone values landed at u_0*u_i, outside the range).
        base = 1.0
        for prim in prims:
            op = scale_op_of(prim)
            if op is not None and op.Get() is not None:
                base = float(op.Get()[0])
                break
        for prim in prims:
            s = base * random.uniform(*scale_range)
            op = scale_op_of(prim) or UsdGeom.Xformable(prim).AddScaleOp()
            op.Set(Gf.Vec3f(s, s, s))


class randomize_robot_color(ManagerTermBase):
    """Per-env jitter of every robot visual material's diffuseColor on reset.

    The robot USD carries per-link UsdPreviewSurface shaders under
    `<Robot>/Looks/*/PBRShader` (authored by scripts/apply_materials.py);
    scene cloning gives each env its own copy, so the jitter is genuinely
    per-env. Keep `variation` <= 0.05: the red link material must not drift
    toward the green HSV band or it spoofs the mask.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # per env: list of (attr, base_color) for every robot material
        self._per_env: list | None = None

    def _resolve(self, env: "ManagerBasedEnv", asset_name: str) -> list:
        import re

        import isaaclab.sim as sim_utils

        pattern = (
            env.scene[asset_name].cfg.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
            + "/Looks/.*/PBRShader"
        )
        per_env = [[] for _ in range(env.num_envs)]
        for prim in sim_utils.find_matching_prims(pattern):
            m = re.search(r"env_(\d+)", str(prim.GetPath()))
            attr = prim.GetAttribute("inputs:diffuseColor")
            if m and attr.IsValid():
                per_env[int(m.group(1))].append((attr, tuple(attr.Get())))
        return per_env

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        variation: float = 0.05,
    ):
        if self._per_env is None:
            self._per_env = self._resolve(env, asset_cfg.name)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        for i in env_ids.tolist():
            for attr, base in self._per_env[i]:
                attr.Set(_balanced_color_jitter(base, variation))
