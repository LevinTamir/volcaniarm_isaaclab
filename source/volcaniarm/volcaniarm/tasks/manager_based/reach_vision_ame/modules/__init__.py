# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""Policy modules for the AME task, registered into rsl_rl's namespace.

`OnPolicyRunner._construct_algorithm` resolves the policy class with a bare
``eval(self.policy_cfg.pop("class_name"))`` evaluated in *rsl_rl's* module
scope. Only names visible there resolve — a dotted path into this package
does not. So we bind the class into `rsl_rl.modules` (which IS in that scope
via the `rsl_rl` import) and onto the runner module directly.

If this injection ever silently fails, training does not crash: rsl_rl falls
back to the stock `ActorCritic`, which would flatten the mask into a 2308-D
MLP input and train something that looks superficially fine. The first
training check is therefore to confirm the constructed class name.
"""

import rsl_rl.modules as _rsl_modules
import rsl_rl.runners.on_policy_runner as _on_policy_runner

from .actor_critic_ame import ActorCriticAME, AmeEncoder  # noqa: F401

_rsl_modules.ActorCriticAME = ActorCriticAME
_on_policy_runner.ActorCriticAME = ActorCriticAME

__all__ = ["ActorCriticAME", "AmeEncoder"]
