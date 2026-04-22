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


def elbow_up_posture(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # Mean knee Z (passive arm_joint origins, i.e. the arm_link body
    # frames) above the robot base. Both 5-bar assembly modes can reach
    # the same EE pose, so the position-tracking reward alone is
    # ambiguous between them — this term breaks the tie toward elbow-up.
    asset: RigidObject = env.scene[asset_cfg.name]
    knee_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    base_z = asset.data.root_pos_w[:, 2:3]
    return (knee_z - base_z).mean(dim=1)


def joint_pos_out_of_range(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, low: float, high: float
) -> torch.Tensor:
    # Linear hinge on each selected joint: max(0, q-high) + max(0, low-q),
    # summed across the selected joints. Zero inside [low, high]; grows
    # linearly outside. Used to keep the actuated 5-bar elbows within a
    # "working range" narrower than the URDF limits, where the linkage is
    # well away from singular configurations.
    asset: RigidObject = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    above = (q - high).clamp(min=0.0)
    below = (low - q).clamp(min=0.0)
    return (above + below).sum(dim=1)
