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
parser.add_argument(
    "--limit-left",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="asymmetric left-elbow sweep bounds (rad); overrides --limit for that joint. "
    "Use for joystick-recorded extents, which are never symmetric.",
)
parser.add_argument(
    "--limit-right",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="asymmetric right-elbow sweep bounds (rad); overrides --limit, see --limit-left.",
)
parser.add_argument(
    "--settle-tol",
    type=float,
    default=0.02,
    help="max |settled - commanded| elbow residual (rad). Samples above this are "
    "discarded: at wide bounds the closed 5-bar can fail to close or flip its "
    "assembly branch, and the FK readout is then garbage that silently inflates "
    "the envelope.",
)
parser.add_argument(
    "--arm-low",
    type=float,
    default=-1.5707963267948966,
    help="passive arm-joint lower bound (rad) for a sample to count as the "
    "working assembly branch (the arm_pos_in_range reward bound).",
)
parser.add_argument(
    "--arm-high",
    type=float,
    default=0.8726646259971648,
    help="passive arm-joint upper bound (rad), see --arm-low.",
)
parser.add_argument(
    "--emit-table",
    nargs="?",
    const="source/volcaniarm/volcaniarm/tasks/manager_based/reach_vision_ame/workspace_table.py",
    default=None,
    help="emit a generated Python module with the z->(y_min,y_max) reachability "
    "table (path optional; default is the AME task package).",
)
parser.add_argument("--table-z-step", type=float, default=0.005, help="table z bin height (m)")
parser.add_argument(
    "--table-z-range",
    type=float,
    nargs=2,
    default=(0.05, 0.30),
    metavar=("Z_LO", "Z_HI"),
    help="world-z range covered by the emitted table (m)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import volcaniarm.tasks  # noqa: F401  -- registers the envs
from isaaclab_tasks.utils import parse_env_cfg

# Elbow sweep bounds. Symmetric ±--limit by default (the reward's in-range
# bound, ±75°); --limit-left/--limit-right override per joint for
# joystick-recorded extents, which are never symmetric.
ELBOW_LIMIT = args_cli.limit
LEFT_BOUNDS = tuple(args_cli.limit_left) if args_cli.limit_left else (-ELBOW_LIMIT, ELBOW_LIMIT)
RIGHT_BOUNDS = tuple(args_cli.limit_right) if args_cli.limit_right else (-ELBOW_LIMIT, ELBOW_LIMIT)
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
    left = torch.linspace(*LEFT_BOUNDS, n, device=dev).repeat_interleave(n)
    right = torch.linspace(*RIGHT_BOUNDS, n, device=dev).repeat(n)

    elbow_ids, elbow_names = robot.find_joints(["volcaniarm_(left|right)_elbow_joint"])
    # find_joints returns sim order; map explicitly so left/right aren't swapped.
    order = [elbow_names.index(nm) for nm in sorted(elbow_names)]
    q = torch.zeros((n * n, len(elbow_ids)), device=dev)
    q[:, order[0]] = left
    q[:, order[1]] = right

    env.reset()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[:, elbow_ids] = q

    # Drive each env from the assembled HOME pose to its commanded elbow
    # angles by ramping the position-drive targets quasi-statically, then
    # hold. This is the only measurement that matches what a policy can do:
    # continuity keeps the closed loop on the WORKING assembly branch.
    # (Teleporting the joints instead lets PhysX settle onto flipped
    # branches — arms folded up over the base — which inflates the envelope
    # with poses the real arm can never reach; and without holding the
    # targets the drives yank everything back home during the settle.)
    ramp_steps, hold_steps = 90, 60
    home = robot.data.default_joint_pos.clone()
    for t in range(ramp_steps + hold_steps):
        alpha = min(1.0, (t + 1) / ramp_steps)
        robot.set_joint_position_target(home + alpha * (joint_pos - home))
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.sim.get_physics_dt())

    ee_id = robot.find_bodies(["left_ee_link"])[0][0]
    ee_w = robot.data.body_pos_w[:, ee_id, :]
    origin = env.scene.env_origins
    ee = ee_w - origin  # per-env local == base-aligned world

    # Validity: (a) the elbows actually settled onto the commanded angles
    # (the loop closed there), AND (b) the passive arms stayed inside the
    # working-branch bound. Ramping keeps almost everything on the working
    # branch, but near-singular ramps can still flip — and a flipped pose
    # is not part of the deployable workspace.
    residual = (robot.data.joint_pos[:, elbow_ids] - joint_pos[:, elbow_ids]).abs()
    settled = (residual < args_cli.settle_tol).all(dim=-1)

    arm_ids, arm_names = robot.find_joints(["volcaniarm_(left|right)_arm_joint"])
    arm_all = robot.data.joint_pos[:, arm_ids]
    on_branch = ((arm_all > args_cli.arm_low) & (arm_all < args_cli.arm_high)).all(dim=-1)

    valid = settled & on_branch
    n_valid = int(valid.sum())

    ee = ee[valid]
    x, y, z = ee[:, 0], ee[:, 1], ee[:, 2]
    zb = z - BASE_Z_WORLD  # base-link frame

    # Passive arm-joint span over the valid sweep — this is what the
    # `arm_pos_in_range` reward bound must cover if the elbow bound widens.
    arm_q = arm_all[valid]

    lines = [
        f"task={args_cli.task}  samples={n*n}  "
        f"elbow L=[{LEFT_BOUNDS[0]:.4f}, {LEFT_BOUNDS[1]:.4f}] "
        f"R=[{RIGHT_BOUNDS[0]:.4f}, {RIGHT_BOUNDS[1]:.4f}] rad",
        f"valid: {n_valid}/{n*n} ({100.0*n_valid/(n*n):.1f}%)  "
        f"[settled<{args_cli.settle_tol:.3f} rad: {int(settled.sum())}; "
        f"arm on-branch ({args_cli.arm_low:.3f},{args_cli.arm_high:.3f}): "
        f"{int(on_branch.sum())}]",
        f"passive arm joints over valid sweep: "
        f"[{arm_q.min():.4f}, {arm_q.max():.4f}] rad",
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

    if args_cli.emit_table:
        lines += emit_table(y, z, arm_q)

    text = "\n".join(lines)
    if args_cli.report:
        with open(args_cli.report, "w") as f:
            f.write(text + "\n")
    else:
        print(text)

    env.close()


def emit_table(y: torch.Tensor, z: torch.Tensor, arm_q: torch.Tensor) -> list[str]:
    """Write the generated z->(y_min, y_max) module; return report lines.

    Bins the *valid* sweep samples by world z. A bin is usable only if it has
    enough samples and its sorted-y max gap is small (the reachable set at
    fixed z is not guaranteed to be a single interval — a hole would otherwise
    make the sampler place targets in unreachable space). Emitted row i is the
    intersection of bins i and i+1, so the interval stays conservative where
    the envelope shrinks with height.
    """
    z_lo_r, z_hi_r = args_cli.table_z_range
    step = args_cli.table_z_step
    n_bins = max(1, round((z_hi_r - z_lo_r) / step))

    report: list[str] = ["", f"emitting table -> {args_cli.emit_table}"]
    bins = []  # (z_lo, z_hi, y_min, y_max, usable)
    for i in range(n_bins):
        lo = z_lo_r + i * step
        hi = lo + step
        m = (z >= lo) & (z < hi)
        count = int(m.sum())
        if count < 8:
            bins.append((lo, hi, 0.0, 0.0, False))
            if count > 0:
                report.append(f"  bin [{lo:.3f},{hi:.3f}): only {count} samples — dropped")
            continue
        ys = torch.sort(y[m]).values
        span = float(ys[-1] - ys[0])
        gaps = (ys[1:] - ys[:-1])
        max_gap = float(gaps.max())
        # Expected spacing if the samples were spread evenly over the span;
        # a gap much larger than that means the set has a hole at this z.
        expected = max(span / (count - 1), 1e-4)
        if max_gap > max(0.010, 5.0 * expected):
            bins.append((lo, hi, 0.0, 0.0, False))
            report.append(
                f"  bin [{lo:.3f},{hi:.3f}): y-hole (max gap {max_gap*100:.1f} cm) — dropped"
            )
            continue
        bins.append((lo, hi, float(ys[0]), float(ys[-1]), True))

    rows = []
    for i, (lo, hi, y0, y1, ok) in enumerate(bins):
        nxt = bins[i + 1] if i + 1 < len(bins) else bins[i]
        if not ok or not nxt[4]:
            continue
        y_min = max(y0, nxt[2])
        y_max = min(y1, nxt[3])
        if y_max - y_min < 0.01:
            report.append(f"  bin [{lo:.3f},{hi:.3f}): intersection with next bin empty — dropped")
            continue
        rows.append((lo, hi, y_min, y_max))

    header = (
        "# Copyright (c) 2026, Tamir Levin.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        '"""AUTO-GENERATED by scripts/check_workspace.py --emit-table -- DO NOT HAND-EDIT.\n'
        "\n"
        "Measured reachable EE envelope of the closed 5-bar, as world-z bins with\n"
        "the y-interval reachable at that height (valid, settled samples only;\n"
        "row i intersected with row i+1 to stay conservative). Regenerate with:\n"
        "\n"
        "    ~/isaac/IsaacLab/isaaclab.sh -p scripts/check_workspace.py \\\n"
        f"        --grid {args_cli.grid} \\\n"
        f"        --limit-left {LEFT_BOUNDS[0]:.10f} {LEFT_BOUNDS[1]:.10f} \\\n"
        f"        --limit-right {RIGHT_BOUNDS[0]:.10f} {RIGHT_BOUNDS[1]:.10f} \\\n"
        f"        --arm-low {args_cli.arm_low:.10f} --arm-high {args_cli.arm_high:.10f} \\\n"
        "        --emit-table\n"
        '"""\n'
        "\n"
        f"ELBOW_BOUNDS_LEFT = ({LEFT_BOUNDS[0]:.10f}, {LEFT_BOUNDS[1]:.10f})\n"
        f"ELBOW_BOUNDS_RIGHT = ({RIGHT_BOUNDS[0]:.10f}, {RIGHT_BOUNDS[1]:.10f})\n"
        f"GRID = {args_cli.grid}\n"
        f"Z_STEP = {step}\n"
        f"ARM_JOINT_RANGE_RAD = ({arm_q.min():.4f}, {arm_q.max():.4f})\n"
        "# rows: (z_lo_world, z_hi_world, y_min, y_max)\n"
    )
    body = "TABLE = [\n"
    for lo, hi, y0, y1 in rows:
        body += f"    ({lo:.3f}, {hi:.3f}, {y0:.4f}, {y1:.4f}),\n"
    body += "]\n"

    with open(args_cli.emit_table, "w") as f:
        f.write(header + body)
    report.append(f"  {len(rows)} usable rows over z=[{z_lo_r}, {z_hi_r}]")
    return report


if __name__ == "__main__":
    main()
    simulation_app.close()
