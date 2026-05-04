# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for the vision reach task.

Re-exports the IsaacLab stdlib mdp + the shared joint-range penalty
from the state-based reach task; adds plant-pose-based tracking rewards
and a privileged plant-pose observation.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from ...reach.mdp.rewards import joint_pos_out_of_range  # noqa: F401
from .rewards import (  # noqa: F401
    plant_pos_in_base,
    position_plant_error,
    position_plant_error_tanh,
)
