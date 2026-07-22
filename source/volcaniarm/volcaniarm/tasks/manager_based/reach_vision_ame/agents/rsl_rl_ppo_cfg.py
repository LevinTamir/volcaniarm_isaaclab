# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""PPO config for the AME vision reach task.

Differs from the `reach_vision` baseline in three ways that matter:

- The policy is `ActorCriticAME` — a CNN over the green mask feeding a single
  spatial-attention layer — instead of an MLP over frozen ResNet18 logits.
- `obs_groups` is set explicitly. It must be: the mask belongs to the actor's
  set and must be *absent* from the critic's, which is what makes the critic
  image-free.
- Observation normalisation is off at this level and handled inside the
  module, on the proprio slice only. See ActorCriticAME for why whitening the
  mask is harmful.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from ..contract import MASK_GROUP, MASK_H, MASK_W

# Imported for its import-time side effect: registering ActorCriticAME into
# rsl_rl's module namespace so the runner's bare eval() can resolve it.
from .. import modules  # noqa: F401


@configclass
class AmeActorCriticCfg(RslRlPpoActorCriticCfg):
    """Extra hyper-parameters passed through to ActorCriticAME.

    `RslRlPpoActorCriticCfg` is a plain configclass, so subclassing is how
    extra fields reach the policy: the runner splats the whole cfg dict as
    **kwargs into the policy constructor.
    """

    class_name: str = "rsl_rl.modules.ActorCriticAME"

    image_obs_group: str = MASK_GROUP
    mask_hw: list[int] = [MASK_H, MASK_W]

    # Second conv must output mha_dim — its channels become the token feature
    # dimension. Stride 2 on BOTH convs is deliberate: at 48x48 with stride 1
    # on the second conv you get 576 tokens, and with a 4096-sample PPO
    # minibatch the attention projections alone run to hundreds of MB each.
    # (2, 2) gives 12x12 = 144 tokens, ample for localising a single blob.
    cnn_channels: list[int] = [16, 64]
    cnn_kernels: list[int] = [5, 3]
    cnn_strides: list[int] = [2, 2]

    # GroupNorm, not AME's BatchNorm. With BatchNorm the rollout runs in
    # train() over 512 correlated same-timestep frames while the PPO update
    # runs on minibatches shuffled across timesteps, so the *same* transition
    # sees different batch statistics. The recomputed log_prob then disagrees
    # with the stored old_actions_log_prob and silently biases the importance
    # ratio. GroupNorm is batch-size independent and identical in
    # train/eval/export. Set "batch" to reproduce AME exactly.
    cnn_norm: str = "group"

    mha_dim: int = 64
    mha_heads: int = 16

    proprio_normalization: bool = True
    critic_uses_image: bool = False


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    # ~3x the old narrow-workspace convergence point (mean reward plateaued
    # at ~iter 450 on 2026-07-20), scaled for the table-driven target space.
    # Stop early when Episode_Reward/weed_position_tracking_tanh_fine
    # flattens — checkpoints land every save_interval anyway.
    max_iterations = 1500
    save_interval = 100
    experiment_name = "volcaniarm_reach_vision_ame"
    empirical_normalization = False

    # REQUIRED. The mask goes to the actor only; the critic gets the
    # privileged low-dim vector and no image.
    obs_groups = {"policy": ["policy", MASK_GROUP], "critic": ["critic"]}

    policy: AmeActorCriticCfg = AmeActorCriticCfg(
        init_noise_std=0.3,
        # Handled inside the module, proprio-only — see the class docstring.
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        # 8 rather than the baseline's 4: the AME encoder's activations scale
        # with token count, so smaller minibatches keep peak memory down.
        num_mini_batches=8,
        # 3e-4 rather than the baseline's 1e-3. That was tuned for a small MLP
        # over frozen features; a conv + attention stack trained end-to-end
        # wants a smaller step, and the adaptive schedule corrects from there.
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Stage2PPORunnerCfg(PPORunnerCfg):
    """Stage-2 fine-tuning runner (task id -v1).

    `experiment_name` is inherited UNCHANGED on purpose: both stages must
    share logs/reach_vision_ame/rsl_rl/volcaniarm_reach_vision_ame/ so
    `--resume --load_run <stage1_run>` (and play/export checkpoint
    discovery) can see stage-1 runs. The run_name suffix is what tells the
    stages apart in TensorBoard. The DR shock will spike KL early on and
    the adaptive schedule will cut the LR — expect ~20 noisy iterations; if
    the policy visibly collapses instead, drop algorithm.learning_rate to
    1e-4 here.
    """

    # Fine-tuning recovers from the DR shock within a few hundred
    # iterations; 1000 leaves comfortable headroom.
    max_iterations = 1000
    run_name = "stage2_dr"
