# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for the AME vision reach task.

Rewards and events are reused verbatim from `reach_vision` by import — the
task differs in how the image is encoded, not in what it is rewarded for.
Keeping them shared is what makes the head-to-head comparison against the
`reach_vision` baseline meaningful.
"""

from ...reach_vision.mdp import *  # noqa: F401, F403

from .green_mask import green_mask, isolate_blob, rgb_to_green_mask  # noqa: F401
