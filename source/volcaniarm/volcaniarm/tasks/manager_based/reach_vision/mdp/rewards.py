# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Reward + observation terms for the vision reach task.

Difference from `reach/mdp/rewards.py`: target position is read from the
plant `RigidObject` in the scene, not from the command manager. The
2-DOF planar arm cannot control X — only Y and Z error are penalised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _yz_distance(curr_pos_w: torch.Tensor, des_pos_w: torch.Tensor) -> torch.Tensor:
    delta = curr_pos_w - des_pos_w
    return torch.norm(delta[:, 1:3], dim=1)


def position_plant_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    plant_cfg: SceneEntityCfg = SceneEntityCfg("plant"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    plant: RigidObject = env.scene[plant_cfg.name]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return _yz_distance(curr_pos_w, plant.data.root_pos_w)


def position_plant_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    plant_cfg: SceneEntityCfg = SceneEntityCfg("plant"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    plant: RigidObject = env.scene[plant_cfg.name]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    distance = _yz_distance(curr_pos_w, plant.data.root_pos_w)
    return 1 - torch.tanh(distance / std)


def plant_pos_in_base(
    env: ManagerBasedRLEnv,
    plant_cfg: SceneEntityCfg = SceneEntityCfg("plant"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Privileged plant position in the robot's base frame.

    Used as a critic-only obs so the value function can converge fast
    while the actor still has to infer position from the image.
    """
    plant: RigidObject = env.scene[plant_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    return plant.data.root_pos_w - robot.data.root_pos_w
