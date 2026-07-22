# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a trained reach policy and record the EE tracking error per step.

The episode-mean `Metrics/ee_pose/position_error` logged during training is
dominated by the transient right after each target resample (targets jump up
to ~1 m away every 4 s). This script records the *instantaneous* error and the
time since the last resample, so the notebook can report the steady-state
(settled) error — the number that actually reflects the arm's precision.

Output: <run>/plots/settled_error.npz   (arrays: error[T,N], time_left[T,N],
        plus scalars resample_period, step_dt)

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p source/volcaniarm/scripts/rsl_rl/eval_settled_error.py \
        --task Volcaniarm-Reach-Play-v0 --num_envs 64 --steps 1200 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record EE tracking error over a policy rollout.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to roll out.")
parser.add_argument("--task", type=str, default="Volcaniarm-Reach-Play-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=1200, help="Number of control steps to record (~40 s at 30 Hz).")
parser.add_argument("--command_name", type=str, default="ee_pose", help="Command term name to read the metric from.")
parser.add_argument(
    "--fractions",
    type=float,
    nargs="+",
    default=[0.01, 0.30, 1.0],
    help="Checkpoints to evaluate, as fractions of the run's max iteration (e.g. 0.05 0.5 1.0). "
    "Nearest available checkpoint is used. Ignored when --checkpoint is given.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

_task_subdir = cli_args.task_log_subdir(args_cli.task)
_hydra_run_dir_override = f"hydra.run.dir=outputs/{_task_subdir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}"
sys.argv = [sys.argv[0], _hydra_run_dir_override] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import os
import re

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import volcaniarm.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    task_subdir = cli_args.task_log_subdir(args_cli.task)
    log_root_path = os.path.abspath(os.path.join("logs", task_subdir, "rsl_rl", agent_cfg.experiment_name))

    # Resolve which checkpoints to evaluate: an explicit --checkpoint, or
    # the checkpoints nearest each requested fraction of the run's length.
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        run_dir = os.path.dirname(resume_path)
        ckpt_paths = [resume_path]
    else:
        latest = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        run_dir = os.path.dirname(latest)
        avail = {}
        for p in glob.glob(os.path.join(run_dir, "model_*.pt")):
            mm = re.search(r"model_(\d+)", os.path.basename(p))
            if mm:
                avail[int(mm.group(1))] = p
        max_it = max(avail)
        chosen = sorted({min(avail, key=lambda it: abs(it - f * max_it)) for f in args_cli.fractions})
        ckpt_paths = [avail[it] for it in chosen]
        print(f"[INFO] Run {os.path.basename(run_dir)}  max_iter={max_it}  "
              f"fractions={args_cli.fractions} -> iters {chosen}")

    # Build the env + runner ONCE; reload weights per checkpoint.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    cmd_term = env.unwrapped.command_manager.get_term(args_cli.command_name)
    resample_period = float(cmd_term.cfg.resampling_time_range[1])
    step_dt = float(env.unwrapped.step_dt)
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for path in ckpt_paths:
        mm = re.search(r"model_(\d+)", os.path.basename(path))
        iteration = int(mm.group(1)) if mm else -1
        runner.load(path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        print(f"[INFO] Rolling out iter {iteration}: {path}")

        errs, tleft = [], []
        # Reset inside inference_mode: the rollout below taints the env
        # buffers as inference tensors, which an out-of-context reset can't
        # update in place.
        with torch.inference_mode():
            reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            errs.append(cmd_term.metrics["position_error"].detach().cpu().numpy().copy())
            tleft.append(cmd_term.time_left.detach().cpu().numpy().copy())

        # One file per checkpoint iteration so the notebook can show how the
        # settled-error distribution evolves over training.
        fname = f"settled_error_iter{iteration:05d}.npz" if iteration >= 0 else "settled_error.npz"
        out = os.path.join(out_dir, fname)
        np.savez(
            out,
            error=np.asarray(errs),          # [T, N], metres
            time_left=np.asarray(tleft),     # [T, N], seconds until next resample
            resample_period=resample_period,  # seconds
            step_dt=step_dt,                  # seconds per control step
            iteration=iteration,              # training iteration of this checkpoint
        )
        print(f"[INFO] Wrote {out}  (shape error={np.asarray(errs).shape})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
