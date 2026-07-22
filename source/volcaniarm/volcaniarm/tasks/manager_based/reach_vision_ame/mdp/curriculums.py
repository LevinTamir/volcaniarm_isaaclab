# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Curriculum terms for the AME vision reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Both weed placement terms must track the same region or the mid-episode
# resample would jump outside the curriculum.
_WEED_EVENT_TERMS = ("randomize_weed_pose", "resample_weed_pose")


def expand_weed_region(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    start_iter: int,
    end_iter: int,
    z_range_start: tuple[float, float],
    z_range_end: tuple[float, float],
    y_scale_start: float,
    y_scale_end: float,
    num_steps_per_env: int = 32,
) -> torch.Tensor:
    """Linearly grow the weed sampling region from a narrow band to the full table.

    The full-region task stalled from a cold start (park-at-centroid optimum,
    2026-07-22 runs); starting at the old validated narrow band reproduces the
    setup that learned in ~50 iterations, then the region expands between
    `start_iter` and `end_iter`. Mutates the params of both weed event terms
    (`z_range` narrows in z, `y_scale` shrinks each table row toward its
    centre — see `reset_weed_in_reachable_workspace`). Returns the progress
    fraction so TensorBoard logs it under Curriculum/.
    """
    iteration = env.common_step_counter // num_steps_per_env
    if iteration <= start_iter:
        progress = 0.0
    elif iteration >= end_iter:
        progress = 1.0
    else:
        progress = (iteration - start_iter) / (end_iter - start_iter)

    z_lo = z_range_start[0] + progress * (z_range_end[0] - z_range_start[0])
    z_hi = z_range_start[1] + progress * (z_range_end[1] - z_range_start[1])
    y_scale = y_scale_start + progress * (y_scale_end - y_scale_start)

    for name in _WEED_EVENT_TERMS:
        term_cfg = env.event_manager.get_term_cfg(name)
        term_cfg.params["z_range"] = (z_lo, z_hi)
        term_cfg.params["y_scale"] = y_scale

    return torch.tensor(progress, device=env.device)
