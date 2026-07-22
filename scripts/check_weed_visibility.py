"""Verify (and optionally clip) the weed sampling table against the camera.

The AME task samples the weed canopy anywhere in the measured reachable
(y, z) envelope (`workspace_table.py`), but nothing in the env constrains the
sample to the camera frustum. A weed outside the 69x42.5 deg FOV — or inside
the 0.195 m near clip, which gets closer as the canopy band rises — renders
to an all-zero mask: the actor goes blind while the critic still sees the
privileged position, and training silently degrades.

This probes a y-grid across every table row (at the row's z midpoint),
renders, and reports which probes produce a usable mask. Modes:

  * default: exit 1 if any margin-shrunk row extreme is invisible — a gate
    to run before training whenever the table, camera rig or band changes.
  * --clip-table: rewrite `workspace_table.py` in place with each row's
    y-interval intersected with the measured visible interval (longest
    contiguous visible run of probes), dropping rows that end up too
    narrow. Turns the table from "reachable" into "reachable AND visible".

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/check_weed_visibility.py [--clip-table]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Volcaniarm-Reach-Vision-AME-Play-v0")
parser.add_argument("--min-mask-sum", type=float, default=2.0,
                    help="min summed 48x48 mask coverage (~full pixels) to count as visible")
parser.add_argument("--grid-per-row", type=int, default=15,
                    help="y probes per table row")
parser.add_argument("--clip-table", action="store_true",
                    help="rewrite workspace_table.py clipped to the visible interval")
parser.add_argument("--report", default=None, help="write report here instead of stdout")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import volcaniarm.tasks  # noqa: F401  -- registers the envs
from isaaclab_tasks.utils import parse_env_cfg

from volcaniarm.tasks.manager_based.reach_vision_ame import workspace_table
from volcaniarm.tasks.manager_based.reach_vision_ame.contract import (
    WEED_X_BASE,
    WEED_Y_SAFETY_MARGIN,
)

TABLE_PATH = workspace_table.__file__
MIN_ROW_WIDTH = 2.0 * WEED_Y_SAFETY_MARGIN + 0.05  # narrower rows are dropped


def main() -> None:
    rows = workspace_table.TABLE
    g = args_cli.grid_per_row
    probes = []  # (row_idx, y, z)
    for r, (z_lo, z_hi, y_min, y_max) in enumerate(rows):
        z_mid = 0.5 * (z_lo + z_hi)
        lo = y_min + WEED_Y_SAFETY_MARGIN
        hi = y_max - WEED_Y_SAFETY_MARGIN
        for k in range(g):
            probes.append((r, lo + (hi - lo) * k / (g - 1), z_mid))

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=len(probes))
    # Deterministic scene: no weed/camera randomisation, no mask corruption.
    for ev in ("randomize_weed_pose", "resample_weed_pose", "randomize_camera_pose",
               "randomize_weed_color", "randomize_dome", "randomize_sun",
               "randomize_mat_color", "reset_robot_joints"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    env_cfg.observations.mask.enable_corruption = False
    if "dropout" in env_cfg.observations.mask.image.params:
        env_cfg.observations.mask.image.params["dropout"] = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    env.reset()
    weed = env.scene["weed"]
    dev = env.device

    y = torch.tensor([p[1] for p in probes], device=dev)
    z = torch.tensor([p[2] for p in probes], device=dev)
    pos = env.scene.env_origins + torch.stack(
        [torch.full_like(y, WEED_X_BASE), y, z], dim=-1
    )
    quat = weed.data.default_root_state[:, 3:7]
    weed.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    weed.write_root_velocity_to_sim(torch.zeros(len(probes), 6, device=dev))

    # A few rendered steps so the TiledCamera picks up the new pose.
    for _ in range(4):
        env.sim.step(render=True)
        env.scene.update(env.sim.get_physics_dt())

    mask = env.observation_manager.compute_group("mask")
    if isinstance(mask, dict):
        mask = next(iter(mask.values()))
    mask_sum = mask.view(len(probes), -1).sum(dim=-1)

    cam = env.scene["base_camera"]
    dist = torch.norm(weed.data.root_pos_w - cam.data.pos_w, dim=-1)
    near_clip = float(env_cfg.scene.base_camera.spawn.clipping_range[0])
    visible = (mask_sum >= args_cli.min_mask_sum) & (dist >= near_clip)

    lines = [
        f"task={args_cli.task}  rows={len(rows)}  probes/row={g}  "
        f"min_mask_sum={args_cli.min_mask_sum}  near_clip={near_clip}",
        f"  {'z_mid':>6} {'y_lo':>8} {'y_hi':>8}  visible probes / interval",
    ]
    clipped = []
    edge_failures = 0
    for r, (z_lo, z_hi, y_min, y_max) in enumerate(rows):
        idx = [i for i, p in enumerate(probes) if p[0] == r]
        vis = [bool(visible[i]) for i in idx]
        ys = [probes[i][1] for i in idx]
        # Longest contiguous visible run.
        best, cur, best_rng, cur_start = 0, 0, None, 0
        for k, v in enumerate(vis):
            if v:
                cur += 1
                if cur == 1:
                    cur_start = k
                if cur > best:
                    best, best_rng = cur, (cur_start, k)
            else:
                cur = 0
        if not vis[0]:
            edge_failures += 1
        if not vis[-1]:
            edge_failures += 1
        z_mid = 0.5 * (z_lo + z_hi)
        if best_rng is None:
            lines.append(f"  {z_mid:>6.3f} {y_min:>8.3f} {y_max:>8.3f}  NONE — row dropped")
            continue
        v_lo, v_hi = ys[best_rng[0]], ys[best_rng[1]]
        # Un-shrink by the sampling margin, but never beyond the reachable row.
        c_lo = max(y_min, v_lo - WEED_Y_SAFETY_MARGIN)
        c_hi = min(y_max, v_hi + WEED_Y_SAFETY_MARGIN)
        status = "full" if best == g else f"clip -> [{c_lo:+.3f}, {c_hi:+.3f}]"
        lines.append(
            f"  {z_mid:>6.3f} {y_min:>8.3f} {y_max:>8.3f}  {best}/{g}  {status}"
        )
        if c_hi - c_lo >= MIN_ROW_WIDTH:
            clipped.append((z_lo, z_hi, c_lo, c_hi))
        else:
            lines.append(f"  {z_mid:>6.3f} row too narrow after clip — dropped")

    if args_cli.clip_table:
        header = (
            "# Copyright (c) 2026, Tamir Levin.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            '"""AUTO-GENERATED -- DO NOT HAND-EDIT.\n'
            "\n"
            "Reachable envelope measured by scripts/check_workspace.py --emit-table,\n"
            "then clipped to the camera-visible interval by\n"
            "scripts/check_weed_visibility.py --clip-table. Regenerate with those two\n"
            "commands in that order (see each script's --help).\n"
            '"""\n'
            "\n"
            f"ELBOW_LIMIT_RAD = {workspace_table.ELBOW_LIMIT_RAD:.10f}\n"
            f"GRID = {workspace_table.GRID}\n"
            f"Z_STEP = {workspace_table.Z_STEP}\n"
            f"ARM_JOINT_RANGE_RAD = {workspace_table.ARM_JOINT_RANGE_RAD}\n"
            "VISIBILITY_CLIPPED = True\n"
            "# rows: (z_lo_world, z_hi_world, y_min, y_max)\n"
        )
        body = "TABLE = [\n"
        for row in clipped:
            body += f"    ({row[0]:.3f}, {row[1]:.3f}, {row[2]:.4f}, {row[3]:.4f}),\n"
        body += "]\n"
        with open(TABLE_PATH, "w") as f:
            f.write(header + body)
        lines.append(f"clipped table written: {len(clipped)}/{len(rows)} rows -> {TABLE_PATH}")

    lines.append(f"{edge_failures} row-extreme visibility failures across {len(rows)} rows")
    text = "\n".join(lines)
    if args_cli.report:
        with open(args_cli.report, "w") as f:
            f.write(text + "\n")
    else:
        print(text)

    env.close()
    if edge_failures and not args_cli.clip_table:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
