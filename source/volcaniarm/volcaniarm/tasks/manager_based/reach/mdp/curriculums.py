# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Curriculum terms for the reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def expand_ee_target_box(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start_iter: int,
    end_iter: int,
    y_start: float,
    y_end: float,
    z_start: tuple,
    z_end: tuple,
    num_steps_per_env: int = 32,
) -> torch.Tensor:
    # Linearly expands the EE target box over training. Before
    # `start_iter` the box stays at the easy "narrow" range. After
    # `end_iter` it stays at the full target range. Between them it
    # interpolates linearly.
    #
    # Iteration is approximated from `env.common_step_counter` and the
    # configured `num_steps_per_env` (rsl_rl rollout length).
    iteration = env.common_step_counter // num_steps_per_env

    if iteration <= start_iter:
        progress = 0.0
    elif iteration >= end_iter:
        progress = 1.0
    else:
        progress = (iteration - start_iter) / (end_iter - start_iter)

    y_half = y_start + progress * (y_end - y_start)
    z_low = z_start[0] + progress * (z_end[0] - z_start[0])
    z_high = z_start[1] + progress * (z_end[1] - z_start[1])

    cmd = env.command_manager.get_term(command_name)
    cmd.cfg.ranges.pos_y = (-y_half, y_half)
    cmd.cfg.ranges.pos_z = (z_low, z_high)

    # Return progress fraction so it's logged as a scalar.
    return torch.tensor(progress, device=env.device)
