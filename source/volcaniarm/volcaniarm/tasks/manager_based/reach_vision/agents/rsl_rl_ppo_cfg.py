# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""PPO config for vision reach.

Note: rsl_rl's stock ActorCritic only accepts 1D obs and ships only
MLPs. We compensate by extracting features upstream — see
`mdp.image_features` with model_name="resnet18". The policy is then a
plain MLP over [features, joint_pos_rel, last_action]; the critic gets
the same plus the privileged plant pose. If frozen ResNet features
turn out too coarse, the fallback is a custom CNN ActorCritic class
registered via `class_name`.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = "volcaniarm_reach_vision"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Wider than the state-based [32, 32] — actor input is now
        # ~1004-D (1000 ResNet logits + joint_pos_rel(2) + last_action(2)).
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Slightly higher than state-based 0.005 — the actor has to
        # explore through visual variation as well as joint space.
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
