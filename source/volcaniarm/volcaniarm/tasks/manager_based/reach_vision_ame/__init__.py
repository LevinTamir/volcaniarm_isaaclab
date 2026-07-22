# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Volcaniarm-Reach-Vision-AME-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.reach_vision_ame_env_cfg:VolcaniarmReachVisionAmeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Stage 2 of the paper-style curriculum: same task, heavier DR, fine-tuned
# from a stage-1 checkpoint. Registered as -v1 ON PURPOSE: task_log_subdir
# strips the -vN suffix, so both stages share
# logs/reach_vision_ame/rsl_rl/volcaniarm_reach_vision_ame/ and the stock
# --resume / play / export checkpoint discovery works across stages.
gym.register(
    id="Volcaniarm-Reach-Vision-AME-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stage2_env_cfg:VolcaniarmReachVisionAmeStage2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2PPORunnerCfg",
    },
)

gym.register(
    id="Volcaniarm-Reach-Vision-AME-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stage2_env_cfg:VolcaniarmReachVisionAmeStage2EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2PPORunnerCfg",
    },
)
