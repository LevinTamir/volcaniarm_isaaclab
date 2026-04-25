# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Custom termination terms for the reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stuck_arm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    dist_threshold: float,
    vel_threshold: float,
    min_episode_steps: int,
) -> torch.Tensor:
    # Terminates an environment when the arm appears to be stuck:
    #   - max actuated joint velocity below `vel_threshold` (not moving),
    #   - AND EE distance from target above `dist_threshold` (not at goal),
    #   - AND past `min_episode_steps` to give the arm time to start
    #     (otherwise the at-rest initial state would trigger termination
    #     on step 0 of every episode).
    # Use `asset_cfg.joint_names` for the actuated joints and
    # `asset_cfg.body_names[0]` for the EE link.
    asset: RigidObject = env.scene[asset_cfg.name]

    past_grace = env.episode_length_buf >= min_episode_steps

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    low_vel = joint_vel.abs().max(dim=1).values < vel_threshold

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    far_from_target = torch.norm(ee_pos_w - des_pos_w, dim=1) > dist_threshold

    return past_grace & low_vel & far_from_target
