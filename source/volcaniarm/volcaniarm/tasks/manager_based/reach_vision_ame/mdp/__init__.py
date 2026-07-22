# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for the AME vision reach task.

Re-exports the IsaacLab stdlib mdp plus the shared joint-range penalty
from the state-based reach task; adds the weed-pose tracking rewards and
camera/light/color randomization events (migrated here from the retired
`reach_vision` ResNet18 baseline) and the green-mask observation.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from ...reach.mdp.rewards import joint_pos_out_of_range  # noqa: F401
from .curriculums import expand_weed_region  # noqa: F401
from .events import (  # noqa: F401
    randomize_camera_pose,
    randomize_light,
    randomize_visual_color_global,
    randomize_weed_color,
    reset_weed_in_reachable_workspace,
)
from .rewards import (  # noqa: F401
    weed_pos_in_base,
    position_weed_error,
    position_weed_error_tanh,
)
from .green_mask import green_mask, isolate_blob, rgb_to_green_mask  # noqa: F401
