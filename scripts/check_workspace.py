"""Map the arm's reachable EE envelope, and report what it can touch at mat height.

The AME task pins the weed to the mat surface (world z=0) and randomises it
along a line in Y. That is only a valid task if the EE can actually get down
to the mat across the sampled Y span — the previous task sampled Z in
[-0.98, -0.78] (base frame), so the mat plane sits at the very bottom edge of
the workspace that was ever exercised.

This sweeps the two actuated elbow joints over their range, reads the true FK
from the physics scene, and reports the reachable (y, z) envelope plus the Y
interval touchable within a tolerance of the mat.

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/check_workspace.py [--report PATH]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# Default to the state-based task: identical robot, no TiledCamera, so the
# sweep needs neither --enable_cameras nor the cost of rendering N^2 cameras.
parser.add_argument("--task", default="Volcaniarm-Reach-v0")
parser.add_argument("--grid", type=int, default=61, help="samples per joint")
parser.add_argument("--tol", type=float, default=0.02, help="mat-contact tolerance (m)")
parser.add_argument(
    "--limit",
    type=float,
    default=1.3089969389957472,
    help="elbow sweep bound (rad). Default is the reward's in-range limit; pass "
    "the articulation limit to see the full mechanical envelope.",
)
parser.add_argument("--report", default=None, help="write report here instead of stdout")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import volcaniarm.tasks  # noqa: F401  -- registers the envs
from isaaclab_tasks.utils import parse_env_cfg

# Elbow limits used by the reward's `elbow_pos_in_range` term (±75°).
# Set from --limit; default is the reward's in-range bound (+-75 deg).
ELBOW_LIMIT = args_cli.limit
BASE_Z_WORLD = 0.98


def main() -> None:
    n = args_cli.grid
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=n * n)
    # Static sweep: strip any randomisation so the commanded joint angles are
    # exactly what we measure. Event sets differ between the state and vision
    # tasks, so clear by name only where present.
    for ev in ("randomize_weed_pose", "resample_weed_pose", "randomize_camera_pose"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    dev = env.device

    # Cartesian product of the two elbow angles.
    sweep = torch.linspace(-ELBOW_LIMIT, ELBOW_LIMIT, n, device=dev)
    left = sweep.repeat_interleave(n)
    right = sweep.repeat(n)

    elbow_ids, elbow_names = robot.find_joints(["volcaniarm_(left|right)_elbow_joint"])
    # find_joints returns sim order; map explicitly so left/right aren't swapped.
    order = [elbow_names.index(nm) for nm in sorted(elbow_names)]
    q = torch.zeros((n * n, len(elbow_ids)), device=dev)
    q[:, order[0]] = left
    q[:, order[1]] = right

    env.reset()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[:, elbow_ids] = q
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.write_data_to_sim()
    # Let the closed 5-bar settle onto the commanded elbow angles.
    for _ in range(4):
        env.sim.step(render=False)
        robot.update(env.sim.get_physics_dt())

    ee_id = robot.find_bodies(["left_ee_link"])[0][0]
    ee_w = robot.data.body_pos_w[:, ee_id, :]
    origin = env.scene.env_origins
    ee = ee_w - origin  # per-env local == base-aligned world

    x, y, z = ee[:, 0], ee[:, 1], ee[:, 2]
    zb = z - BASE_Z_WORLD  # base-link frame

    lines = [
        f"task={args_cli.task}  samples={n*n}  elbow range=+-{ELBOW_LIMIT:.4f} rad",
        f"EE x : [{x.min():.4f}, {x.max():.4f}]  (world)",
        f"EE y : [{y.min():.4f}, {y.max():.4f}]  (world)",
        f"EE z : [{z.min():.4f}, {z.max():.4f}]  (world)   base-frame z: [{zb.min():.4f}, {zb.max():.4f}]",
        "",
        f"mat plane is world z=0  ->  base-frame z={-BASE_Z_WORLD:.4f}",
        f"lowest EE reaches world z={z.min():.4f}  (gap to mat: {z.min():.4f} m)",
        "",
    ]

    near = z <= (z.min() + args_cli.tol)
    if near.any():
        lines.append(
            f"within {args_cli.tol*100:.0f} cm of the LOWEST reachable z, "
            f"y spans [{y[near].min():.4f}, {y[near].max():.4f}]"
        )
    for tol in (0.02, 0.05, 0.10):
        m = z <= tol
        if m.any():
            lines.append(
                f"EE can reach world z<={tol:.2f} m at y in "
                f"[{y[m].min():.4f}, {y[m].max():.4f}]  ({int(m.sum())} configs)"
            )
        else:
            lines.append(f"EE can NEVER reach world z<={tol:.2f} m")

    # Reachable Y span as a function of height — this is what decides where a
    # ground-standing weed's canopy can be, and how wide the Y randomisation
    # can go at that height.
    lines += ["", "reachable Y span per height band (world z):"]
    lines.append(f"  {'z_lo':>6} {'z_hi':>6} {'y_min':>8} {'y_max':>8} {'span':>7} {'n':>5}")
    edges = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for lo_z, hi_z in zip(edges[:-1], edges[1:]):
        m = (z >= lo_z) & (z < hi_z)
        if m.any():
            lines.append(
                f"  {lo_z:>6.3f} {hi_z:>6.3f} {y[m].min():>8.4f} {y[m].max():>8.4f} "
                f"{y[m].max()-y[m].min():>7.4f} {int(m.sum()):>5}"
            )
        else:
            lines.append(f"  {lo_z:>6.3f} {hi_z:>6.3f} {'—':>8} {'—':>8} {'—':>7} {0:>5}")

    # The sweep bound above is the reward's in-range limit, not necessarily the
    # articulation's. If the real limits are wider the true envelope is larger.
    lim = robot.data.joint_pos_limits[0, elbow_ids, :]
    lines += ["", "actual articulation elbow limits (rad):"]
    for nm, i in zip(elbow_names, range(len(elbow_ids))):
        lines.append(f"  {nm}: [{lim[i,0]:.4f}, {lim[i,1]:.4f}]   (swept +-{ELBOW_LIMIT:.4f})")

    text = "\n".join(lines)
    if args_cli.report:
        with open(args_cli.report, "w") as f:
            f.write(text + "\n")
    else:
        print(text)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
