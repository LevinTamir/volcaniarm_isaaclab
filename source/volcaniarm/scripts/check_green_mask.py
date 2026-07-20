# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Visual smoke test for the AME green mask.

Dumps `rgb | mask | isolated` side by side as PNGs so the segmentation can be
eyeballed before committing a training run to it. Numbers alone don't catch a
mask that is bright but centred on the wrong object.

It also reports the measured HSV distribution over green-dominant pixels,
which is what `contract.GREEN_NOMINAL` should be calibrated against — the
material colour is not what lands in the frame once lighting and tonemapping
are applied.

Usage:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p source/volcaniarm/scripts/check_green_mask.py \\
        --num_envs 4 --steps 60 --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Volcaniarm-Reach-Vision-AME-Play-v0")
parser.add_argument("--num_envs", type=int, default=4, help="Envs (one PNG row each).")
parser.add_argument("--steps", type=int, default=60, help="Steps before capture.")
parser.add_argument(
    "--resets", type=int, default=3, help="Capture after this many resets (exercises DR)."
)
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import volcaniarm.tasks  # noqa: F401  -- registers the envs
from isaaclab_tasks.utils import parse_env_cfg
from volcaniarm.tasks.manager_based.reach_vision_ame.contract import MASK_H, MASK_W
from volcaniarm.tasks.manager_based.reach_vision_ame.mdp.green_mask import (
    isolate_blob,
    rgb_to_green_mask,
)


def _to_rgb_u8(mask: torch.Tensor) -> np.ndarray:
    """Grayscale mask -> upscaled uint8 RGB, nearest-neighbour."""
    m = (mask.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    m = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)  # 48 -> 96
    return np.stack([m, m, m], axis=-1)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    action_dim = env.action_space.shape[-1]
    zero = torch.zeros(env.num_envs, action_dim, device=env.device)

    env.reset()
    for _ in range(max(args_cli.resets, 1)):
        env.reset()
        for _ in range(args_cli.steps):
            env.step(zero)

    rgb = env.scene.sensors["base_camera"].data.output["rgb"]
    mask = rgb_to_green_mask(rgb)
    iso = isolate_blob(mask)

    out_dir = args_cli.output_dir or os.path.abspath(
        os.path.join("logs", "reach_vision_ame", "green_mask")
    )
    os.makedirs(out_dir, exist_ok=True)

    report = []
    for i in range(rgb.shape[0]):
        panel = np.concatenate(
            [rgb[i].cpu().numpy(), _to_rgb_u8(mask[i]), _to_rgb_u8(iso[i])], axis=1
        )
        path = os.path.join(out_dir, f"env_{i:02d}.png")
        Image.fromarray(panel).save(path)

        bg = iso[i][iso[i] < 0.5]
        report.append(
            f"env {i}: blob_peak={float(iso[i].max()):.4f} "
            f"coverage={float((iso[i] > 0.5).float().mean()):.4f} "
            f"bg_mean={float(bg.mean()) if bg.numel() else 0.0:.5f} "
            f"dropped_by_isolation={float((mask[i] > 0.5).float().mean() - (iso[i] > 0.5).float().mean()):.5f}"
        )

    # HSV over green-dominant pixels — calibrate GREEN_NOMINAL against this.
    x = rgb.permute(0, 3, 1, 2).float() / 255.0
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx, mn = x.max(dim=1).values, x.min(dim=1).values
    d, eps = mx - mn, 1e-6
    h_r = torch.where(g >= b, (g - b) / (d + eps), (g - b) / (d + eps) + 6.0)
    h = torch.where(mx == r, h_r, torch.where(mx == g, (b - r) / (d + eps) + 2.0, (r - g) / (d + eps) + 4.0)) / 6.0
    h = torch.where(d < eps, torch.zeros_like(h), h)
    s, v = d / (mx + eps), mx
    sel = (g > r * 1.25) & (g > b * 1.15) & (s > 0.25) & (v > 0.15)
    if int(sel.sum()) > 0:
        qs = torch.tensor([0.05, 0.5, 0.95], device=x.device)
        for nm, t in (("hue", h), ("sat", s), ("val", v)):
            q = torch.quantile(t[sel].float(), qs)
            report.append(f"{nm}: p05={q[0]:.4f} median={q[1]:.4f} p95={q[2]:.4f}")
        report.append(f"OpenCV H median = {float(torch.median(h[sel])) * 180:.1f}")
    else:
        report.append("WARNING: no green-dominant pixels found — is the weed in frame?")

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"[check_green_mask] wrote {rgb.shape[0]} PNGs + report.txt to {out_dir}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
