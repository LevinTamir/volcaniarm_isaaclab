# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

##
# Register Gym environments.
##
# NOTE: `rsl_rl_cfg_entry_point` is intentionally absent until the AME
# actor-critic lands — the env is loadable and viewable now (zero_agent,
# render_check), it just cannot be trained yet.

gym.register(
    id="Volcaniarm-Reach-Vision-AME-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.reach_vision_ame_env_cfg:VolcaniarmReachVisionAmeEnvCfg"
        ),
    },
)

gym.register(
    id="Volcaniarm-Reach-Vision-AME-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.reach_vision_ame_env_cfg:VolcaniarmReachVisionAmeEnvCfg_PLAY"
        ),
    },
)
