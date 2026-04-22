# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0

"""ArticulationCfg for the volcaniarm 5-bar planar 2-DOF arm.

Sources the closed-loop USD built by `scripts/close_loop.py`
(sublayers `volcaniarm.usd` and adds `closure_joint`).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "usd").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repo root (with assets/usd/) from {here}")


_REPO_ROOT = _find_repo_root()
_USD_PATH = str(_REPO_ROOT / "assets" / "usd" / "volcaniarm_closed.usd")


VOLCANIARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Lift `base_link` so the caster wheels sit on the ground plane.
        # 0.98 = -min(z) across all cart + caster-wheel meshes.
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            # Centered inside the mechanical limits (right: [π/4, 8π/9],
            # left: [-8π/9, -π/4]). Sets the arm in its natural splayed
            # configuration so the closure constraint resolves cleanly.
            "volcaniarm_left_elbow_joint": -1.5707963267948966,   # -π/2
            "volcaniarm_right_elbow_joint": 1.5707963267948966,   # +π/2
            "volcaniarm_left_arm_joint": 0.0,
            "volcaniarm_right_arm_joint": 0.0,
        },
    ),
    actuators={
        "elbows": ImplicitActuatorCfg(
            joint_names_expr=["volcaniarm_(left|right)_elbow_joint"],
            stiffness=400.0,
            damping=40.0,
        ),
        # Override the 400/40 drive the URDF→USD converter baked in for
        # every joint: these two are passive in the 5-bar loop — no motor.
        "arms_passive": ImplicitActuatorCfg(
            joint_names_expr=["volcaniarm_(left|right)_arm_joint"],
            stiffness=0.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
