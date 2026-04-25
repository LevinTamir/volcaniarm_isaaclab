# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for the reach task.

Vendored from `isaaclab_tasks.manager_based.manipulation.reach.mdp` so
this task stays self-contained. Orientation reward omitted — the 5-bar
is planar and EE orientation isn't independently controllable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 distance (world frame) between `left_ee_link` and the sampled target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Smooth shaping reward: `1 - tanh(d / std)`."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def joint_pos_out_of_range(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, low: float, high: float
) -> torch.Tensor:
    # Linear hinge per selected joint: max(0, q-high) + max(0, low-q),
    # summed across joints. Zero inside [low, high]; grows linearly
    # outside. URDF keeps wide ±π limits so the closure-constraint
    # solver stays stable; this soft penalty teaches the policy to stay
    # within the mechanical operating range (±65° measured in Isaac Sim).
    asset: RigidObject = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    above = (q - high).clamp(min=0.0)
    below = (low - q).clamp(min=0.0)
    return (above + below).sum(dim=1)
