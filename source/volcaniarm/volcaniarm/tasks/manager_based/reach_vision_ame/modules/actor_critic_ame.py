# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""AME actor-critic: CNN over the green mask + spatial attention.

Port of "Attention-Based Map Encoding for Learning Generalized Legged
Locomotion" (SII-FUSC/AME_Locomotion). AME's attention is **spatial, not
temporal**: a small CNN turns a structured map into a grid of spatial tokens,
proprioception is embedded into a *single query token*, and one multi-head
attention layer lets that query attend over the map. There is no sequence
axis and no frame stacking.

Our map is the green-coverage mask rather than an elevation map. AME's key
deviation is feeding its CNN xyz coordinates instead of z-only height —
geometry, not bare sensor values — so the analogue here is `(mask, u, v)`:
the mask plus two normalised pixel-coordinate channels. That lands on three
input channels, so AME's `Conv2d(3->16, k=5)` first layer carries over
unchanged, and because `u,v` are a constant grid they live in this module
rather than in the rollout buffer.
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class AmeEncoder(nn.Module):
    """CNN -> spatial tokens, attended by a single proprio-derived query."""

    def __init__(
        self,
        num_proprio: int,
        mask_hw: tuple[int, int],
        cnn_channels: tuple[int, int] = (16, 64),
        cnn_kernels: tuple[int, int] = (5, 3),
        cnn_strides: tuple[int, int] = (2, 2),
        cnn_norm: str = "group",
        mha_dim: int = 64,
        mha_heads: int = 16,
    ) -> None:
        super().__init__()
        h, w = mask_hw
        self.mask_hw = (h, w)
        c1, c2 = cnn_channels
        if c2 != mha_dim:
            raise ValueError(
                f"second conv must output mha_dim so its channels become token "
                f"features (got {c2} vs mha_dim={mha_dim})"
            )

        def make_norm(c: int) -> nn.Module:
            if cnn_norm == "batch":
                return nn.BatchNorm2d(c)
            if cnn_norm == "group":
                return nn.GroupNorm(min(8, c), c)
            if cnn_norm == "none":
                return nn.Identity()
            raise ValueError(f"unknown cnn_norm: {cnn_norm}")

        # Conv -> ReLU -> Norm ordering is AME's, kept for faithfulness.
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, cnn_kernels[0], stride=cnn_strides[0], padding=cnn_kernels[0] // 2),
            nn.ReLU(),
            make_norm(c1),
            nn.Conv2d(c1, c2, cnn_kernels[1], stride=cnn_strides[1], padding=cnn_kernels[1] // 2),
            nn.ReLU(),
            make_norm(c2),
        )

        self.query = nn.Linear(num_proprio, mha_dim)
        self.mha = nn.MultiheadAttention(mha_dim, mha_heads, batch_first=True)
        self.out_dim = mha_dim

        # Constant (u, v) grid — AME's "xyz instead of z". Registered as a
        # buffer so it moves with .to(device) and is emitted as an ONNX
        # initializer, never stored per-sample in the rollout buffer.
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij"
        )
        self.register_buffer("coord_grid", torch.stack([xx, yy], dim=0).unsqueeze(0))

    def forward(self, mask_flat: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        b = mask_flat.shape[0]
        m = mask_flat.view(b, 1, *self.mask_hw)
        x = torch.cat([m, self.coord_grid.expand(b, -1, -1, -1)], dim=1)
        feat = self.conv(x)                                   # (B, D, h', w')
        tokens = feat.flatten(2).transpose(1, 2)              # (B, h'*w', D)
        q = self.query(proprio).unsqueeze(1)                  # (B, 1, D)
        # need_weights=False keeps the fast path off during export and avoids
        # materialising the attention matrix.
        attended, _ = self.mha(q, tokens, tokens, need_weights=False)
        return attended.squeeze(1)


class ActorCriticAME(nn.Module):
    """rsl_rl-compatible actor-critic with an AME encoder on the actor.

    The critic is deliberately **image-free**: its observation group already
    carries `weed_pos_in_base`, the exact target. Adding a vision encoder on
    top of ground truth would roughly double encoder cost and make the value
    loss compete with the policy gradient over shared conv weights. Set
    `critic_uses_image=True` to A/B that choice.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        image_obs_group: str = "mask",
        mask_hw: tuple[int, int] | list[int] = (48, 48),
        cnn_channels: tuple[int, int] | list[int] = (16, 64),
        cnn_kernels: tuple[int, int] | list[int] = (5, 3),
        cnn_strides: tuple[int, int] | list[int] = (2, 2),
        cnn_norm: str = "group",
        mha_dim: int = 64,
        mha_heads: int = 16,
        proprio_normalization: bool = True,
        critic_uses_image: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticAME.__init__ got unexpected arguments, which will be "
                "ignored: " + str(list(kwargs))
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.image_group = image_obs_group
        self.mask_hw = tuple(mask_hw)
        self.critic_uses_image = critic_uses_image
        self.state_dependent_std = state_dependent_std

        # Proprio groups = everything in the set except the mask.
        self.actor_proprio_groups = [g for g in obs_groups["policy"] if g != image_obs_group]
        self.critic_proprio_groups = [g for g in obs_groups["critic"] if g != image_obs_group]
        if image_obs_group not in obs_groups["policy"]:
            raise ValueError(
                f"obs_groups['policy'] must contain '{image_obs_group}'; got "
                f"{obs_groups['policy']}"
            )

        n_actor_proprio = sum(obs[g].shape[-1] for g in self.actor_proprio_groups)
        n_critic_proprio = sum(obs[g].shape[-1] for g in self.critic_proprio_groups)
        n_mask = obs[image_obs_group].shape[-1]
        if n_mask != self.mask_hw[0] * self.mask_hw[1]:
            raise ValueError(
                f"mask group '{image_obs_group}' has {n_mask} elements but "
                f"mask_hw={self.mask_hw} implies {self.mask_hw[0] * self.mask_hw[1]}"
            )

        self.actor_encoder = AmeEncoder(
            n_actor_proprio, self.mask_hw, tuple(cnn_channels), tuple(cnn_kernels),
            tuple(cnn_strides), cnn_norm, mha_dim, mha_heads,
        )
        actor_in = self.actor_encoder.out_dim + n_actor_proprio
        actor_out = [2, num_actions] if state_dependent_std else num_actions
        self.actor = MLP(actor_in, actor_out, actor_hidden_dims, activation)

        if critic_uses_image:
            self.critic_encoder = AmeEncoder(
                n_critic_proprio, self.mask_hw, tuple(cnn_channels), tuple(cnn_kernels),
                tuple(cnn_strides), cnn_norm, mha_dim, mha_heads,
            )
            critic_in = self.critic_encoder.out_dim + n_critic_proprio
        else:
            self.critic_encoder = None
            critic_in = n_critic_proprio
        self.critic = MLP(critic_in, 1, critic_hidden_dims, activation)

        print(f"AME actor encoder: {self.actor_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Normalisation is applied to the PROPRIO SLICE ONLY. Whitening the
        # mask would be actively harmful: it is ~98% zeros, so its running std
        # collapses toward eps and the normaliser would amplify pixel noise by
        # orders of magnitude. The mask is already in [0,1] and is exactly what
        # the ONNX graph produces from a real frame, so it passes through raw.
        # `actor_obs_normalization` / `critic_obs_normalization` are accepted
        # for signature compatibility but intentionally not used for the mask.
        self.proprio_normalization = proprio_normalization
        if proprio_normalization:
            self.actor_proprio_normalizer = EmpiricalNormalization(n_actor_proprio)
            self.critic_obs_normalizer = EmpiricalNormalization(n_critic_proprio)
        else:
            self.actor_proprio_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if not state_dependent_std:
            if noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"unknown noise_std_type: {noise_std_type}")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # -- observation plumbing -------------------------------------------------

    def _cat(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[g] for g in groups], dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._cat(obs, self.actor_proprio_groups)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._cat(obs, self.critic_proprio_groups)

    def actor_forward(self, proprio: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
        """The ONLY path from (proprio, mask) to an action mean.

        Training and the ONNX bundle both call this, so sim/deploy parity is
        structural rather than a convention someone has to remember.
        """
        p = self.actor_proprio_normalizer(proprio)
        z = self.actor_encoder(mask_flat, p)
        return self.actor(torch.cat([z, p], dim=-1))

    # -- rsl_rl API -----------------------------------------------------------

    def _update_distribution(self, obs: TensorDict) -> None:
        out = self.actor_forward(self.get_actor_obs(obs), obs[self.image_group])
        if self.state_dependent_std:
            mean, std = out[..., 0, :], out[..., 1, :]
            std = torch.exp(std) if self.noise_std_type == "log" else std
        else:
            mean = out
            std = (
                self.std.expand_as(mean)
                if self.noise_std_type == "scalar"
                else torch.exp(self.log_std).expand_as(mean)
            )
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        out = self.actor_forward(self.get_actor_obs(obs), obs[self.image_group])
        return out[..., 0, :] if self.state_dependent_std else out

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        c = self.critic_obs_normalizer(self.get_critic_obs(obs))
        if self.critic_encoder is None:
            return self.critic(c)
        z = self.critic_encoder(obs[self.image_group], c)
        return self.critic(torch.cat([z, c], dim=-1))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.proprio_normalization:
            self.actor_proprio_normalizer.update(self.get_actor_obs(obs))
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError("use act / act_inference / evaluate")

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        # MUST return True. OnPolicyRunner uses the return value to decide
        # whether this is a resumed run; returning None silently discards the
        # optimizer state and resets the iteration counter.
        super().load_state_dict(state_dict, strict=strict)
        return True
