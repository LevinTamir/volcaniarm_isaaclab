# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Render-frame inspection for vision tasks.

Spawns a few envs of the requested task, resets once, lets the camera
render a couple of frames, then dumps the RGB output as PNG so the
camera offset / FOV / weed visibility can be verified before training.

Usage:
    isaaclab.sh -p source/volcaniarm/scripts/render_check.py \\
        --task Volcaniarm-Reach-Vision-AME-v0 [--num_envs 4]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump base-camera frames for visual inspection.")
parser.add_argument("--task", type=str, default="Volcaniarm-Reach-Vision-AME-v0", help="Task to instantiate.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs (one PNG per env).")
parser.add_argument(
    "--sensor", type=str, default="base_camera", help="Scene sensor name to dump."
)
parser.add_argument(
    "--warmup_steps", type=int, default=2, help="Zero-action steps to let the renderer warm up."
)
parser.add_argument("--output_dir", type=str, default=None, help="Override output dir.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras have to be enabled for TiledCamera to actually render.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import volcaniarm.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # Build cfg via the IsaacLab helper so the task's cfg class is
    # resolved correctly without us hard-coding the import.
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[render_check] env created: {type(env).__name__}", flush=True)

    obs, _ = env.reset()
    print(f"[render_check] reset OK; policy obs shape={tuple(obs['policy'].shape)}", flush=True)

    # A few zero-action steps so the renderer settles. Action dim = 2.
    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)
    for _ in range(args_cli.warmup_steps):
        env.step(zero_action)

    # Pull the RGB tensor straight off the sensor — same data path the
    # `mdp.image` obs term uses.
    sensor = env.unwrapped.scene.sensors[args_cli.sensor]
    rgb = sensor.data.output["rgb"]  # (num_envs, H, W, 3) or (num_envs, H, W, 4)
    print(f"[render_check] {args_cli.sensor} rgb tensor: shape={tuple(rgb.shape)} "
          f"dtype={rgb.dtype} device={rgb.device}", flush=True)

    rgb_np = rgb.detach().cpu().numpy()
    if rgb_np.dtype != np.uint8:
        # Some pipelines return float in [0, 1]; clamp + scale.
        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        rgb_np = (rgb_np * 255).astype(np.uint8)

    # Drop any alpha channel.
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]

    # Output dir: logs/<task>/render_check/ by default — mirrors the
    # logs layout written by train.py.
    if args_cli.output_dir:
        out_dir = args_cli.output_dir
    else:
        # local import to avoid any circular issues
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsl_rl"))
        from cli_args import task_log_subdir  # noqa: E402

        task_subdir = task_log_subdir(args_cli.task)
        out_dir = os.path.abspath(os.path.join("logs", task_subdir, "render_check"))
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for i in range(rgb_np.shape[0]):
        path = os.path.join(out_dir, f"env_{i:02d}.png")
        Image.fromarray(rgb_np[i]).save(path)
        written.append(path)

    print(f"[render_check] wrote {len(written)} PNGs to {out_dir}:", flush=True)
    for p in written:
        print(f"  {p}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
