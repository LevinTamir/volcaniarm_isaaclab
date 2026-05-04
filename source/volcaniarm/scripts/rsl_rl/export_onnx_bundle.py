# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Export a trained vision-reach policy as a single bundled ONNX.

The stock `isaaclab_rl.rsl_rl.export_policy_as_onnx` only ships the
actor MLP; it expects a flat 1004-dim observation that already includes
the 1000-dim ResNet18 logits. That contract doesn't match the
deployment side (the ros2_control plugin will receive a raw RGB image,
not pre-extracted features), so we wrap

    raw image (uint8, NHWC)
      → permute + scale + ImageNet-normalize
      → ResNet18 (frozen, IMAGENET1K_V1)
      → concat with [joint_pos_rel, last_action]
      → actor's EmpiricalNormalization
      → actor MLP

into a single `torch.nn.Module` and export that. The deployed
controller then runs a single ONNX session with three named inputs
(`image`, `joint_pos_rel`, `last_action`) and one output (`action`).

Usage:
    isaaclab.sh -p source/volcaniarm/scripts/rsl_rl/export_onnx_bundle.py \\
        --task Volcaniarm-Reach-Vision-Play-v0 \\
        [--load_run <run_name>] [--checkpoint model_<n>.pt]
"""

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export bundled ONNX (image+proprio → action).")
parser.add_argument("--task", type=str, default="Volcaniarm-Reach-Vision-Play-v0", help="Task to load cfg from.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Envs needed only to construct the rsl_rl runner; not used for inference.",
)
parser.add_argument(
    "--output_filename", type=str, default="policy_bundled.onnx", help="ONNX filename inside the run's exported/ dir."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Cameras must be enabled: the env construction below spawns a TiledCamera.
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from torchvision.models import ResNet18_Weights, resnet18

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import volcaniarm.tasks  # noqa: F401


class BundledVisionPolicy(torch.nn.Module):
    """Image + proprio → action, in one ONNX-friendly module."""

    # ImageNet normalization constants — match
    # isaaclab.envs.mdp.observations.image_features._prepare_resnet_model
    # exactly so the bundled ONNX produces identical features to the
    # in-process IsaacLab pipeline.
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, actor: torch.nn.Module, normalizer: torch.nn.Module):
        super().__init__()
        # Load fresh ImageNet ResNet18, weights frozen.
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        for p in self.resnet.parameters():
            p.requires_grad_(False)
        self.normalizer = copy.deepcopy(normalizer).eval()
        self.actor = copy.deepcopy(actor).eval()
        for p in self.normalizer.parameters():
            p.requires_grad_(False)
        for p in self.actor.parameters():
            p.requires_grad_(False)
        self.register_buffer("imagenet_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1))

    def forward(
        self,
        image: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        # image: (N, H, W, 3) uint8 (matches what the IsaacLab obs term
        # produces and what cv_bridge will hand the deployed controller).
        x = image.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.imagenet_mean) / self.imagenet_std
        features = self.resnet(x)
        # Order MUST match `ObservationsCfg.PolicyCfg` term declaration
        # order in reach_vision_env_cfg.py:
        #   joint_pos, actions, image_features  →  [jpr(2), last_action(2), features(1000)].
        obs = torch.cat([joint_pos_rel, last_action, features], dim=1)
        normed = self.normalizer(obs)
        return self.actor(normed)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Resolve the trained checkpoint via the same helper play.py uses.
    task_subdir = cli_args.task_log_subdir(args_cli.task)
    log_root_path = os.path.abspath(
        os.path.join("logs", task_subdir, "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[export_onnx_bundle] log root: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[export_onnx_bundle] loading checkpoint: {resume_path}")

    # Spin up the env just long enough to construct the rsl_rl runner.
    # The runner needs the obs/action shapes it pulls from the env spec.
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # rsl_rl ≥ 2.3 exposes `.alg.policy`; older `.alg.actor_critic`.
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    actor_mlp = policy_nn.actor
    actor_normalizer = getattr(policy_nn, "actor_obs_normalizer", torch.nn.Identity())

    bundled = BundledVisionPolicy(actor_mlp, actor_normalizer).cpu().eval()

    # Pull resolution off the live camera sensor so we never drift from
    # the env_cfg constants.
    sensor = env.unwrapped.scene.sensors["base_camera"]
    H, W = int(sensor.cfg.height), int(sensor.cfg.width)
    print(f"[export_onnx_bundle] camera resolution: {W}x{H}")

    image_dummy = torch.zeros(1, H, W, 3, dtype=torch.uint8)
    jpr_dummy = torch.zeros(1, 2)
    la_dummy = torch.zeros(1, 2)

    out_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args_cli.output_filename)

    torch.onnx.export(
        bundled,
        (image_dummy, jpr_dummy, la_dummy),
        out_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image", "joint_pos_rel", "last_action"],
        output_names=["action"],
        dynamic_axes={
            "image": {0: "batch"},
            "joint_pos_rel": {0: "batch"},
            "last_action": {0: "batch"},
            "action": {0: "batch"},
        },
    )
    print(f"[export_onnx_bundle] wrote {out_path}")

    # Smoke-test: load with onnxruntime and run a dummy inference.
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_onnx_bundle] onnxruntime not installed — skipping numerical check.")
        env.close()
        return

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_action = sess.run(
        None,
        {
            "image": image_dummy.numpy(),
            "joint_pos_rel": jpr_dummy.numpy(),
            "last_action": la_dummy.numpy(),
        },
    )[0]
    print(f"[export_onnx_bundle] onnxruntime action: shape={onnx_action.shape} finite={np.isfinite(onnx_action).all()}")

    # Numerical agreement: torch path on the same dummy buffers.
    with torch.inference_mode():
        torch_action = bundled(image_dummy, jpr_dummy, la_dummy).numpy()
    diff = np.max(np.abs(torch_action - onnx_action))
    print(f"[export_onnx_bundle] |torch − onnx| max-abs diff: {diff:.6e}")
    if diff > 1e-4:
        print("[export_onnx_bundle] WARNING: bundled-ONNX numerics drift > 1e-4 — investigate before deploy.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
