#!/usr/bin/env python3
"""Generate the Isaac-specific URDF from the canonical volcaniarm_ws xacro.

The URDF at assets/urdf/urdf/volcaniarm.urdf used to be a hand-maintained
fork of the ROS2 workspace's volcaniarm.urdf.xacro, and it drifted. This
script makes it a build artifact instead: it runs xacro on the ws model and
applies the Isaac-specific transforms programmatically, so an edit to the ws
xacro is one `sync_urdf.py` run away from the Isaac side.

Transforms applied to the flattened ws URDF:
  1. Strip <gazebo>, <ros2_control>, <transmission> blocks (simulator- and
     control-stack-specific; the Isaac importer must not see them).
  2. Strip frame-only helper links that must not become PhysX articulation
     bodies: `world` (+ world_to_base — build_lab.py places the robot) and
     the four RealSense TF frames (camera_depth_frame,
     camera_depth_optical_frame, camera_color_frame,
     camera_color_optical_frame — published by robot_state_publisher on the
     ROS side; Isaac's camera prim carries its own frame).
  3. Remove the Gazebo closure scaffolding — `right_arm_tip_link` (green
     debug sphere) and `closure_dummy_link` (purple sphere on the revolute
     closure_joint) — after capturing their joint origins.
  4. Synthesize `left_ee_link` / `right_ee_link` (frame-only, mass 0.05,
     inertia 1e-5) at the captured closure point on each arm. close_loop.py
     and every reach task reference these names; the closure_joint proper is
     authored in USD by close_loop.py (URDF cannot express kinematic loops).
  5. Rewrite package:// mesh URIs to absolute paths into this repo's mesh
     copy (the Isaac URDF importer cannot resolve package:// without a ROS
     environment) and hash-sync the STLs from the ws.

Cross-checks (hard failures):
  * ws closure_joint rpy-x must equal CLOSURE_RPY_X in scripts/close_loop.py.
  * Revolute joint document order must be [left_elbow, left_arm,
    right_elbow, right_arm] — the joint-order contract shared with
    volcaniarm_controllers and the trained policies.
  * Final link set must match EXPECTED_LINKS exactly.

Usage:
    python3 scripts/sync_urdf.py                 # regenerate URDF + meshes
    python3 scripts/sync_urdf.py --check         # exit 1 if committed URDF is stale
    python3 scripts/sync_urdf.py --ws ~/other_ws --skip-meshes

Plain python3 — no Isaac imports; needs a *built* volcaniarm_ws (xacro
resolves $(find volcaniarm_description) against the install space).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_URDF = PROJECT / "assets/urdf/urdf/volcaniarm.urdf"
MESH_DIR = PROJECT / "assets/urdf/meshes"
CLOSE_LOOP = PROJECT / "scripts/close_loop.py"

XACRO_REL = "src/volcaniarm_description/urdf/volcaniarm.urdf.xacro"
MESH_SRC_REL = "src/volcaniarm_description/meshes"
# mode:=work is required: other modes add apriltag/calibration links that
# have no business in the training articulation (verify below catches it).
XACRO_ARGS = ["mode:=work", "calibration:=false"]

STRIP_BLOCKS = ("gazebo", "ros2_control", "transmission")
STRIP_LINKS = (
    "world",
    "camera_depth_frame",
    "camera_depth_optical_frame",
    "camera_color_frame",
    "camera_color_optical_frame",
)
DEBUG_LINKS = ("right_arm_tip_link", "closure_dummy_link")

EE_MASS = "0.05"
EE_INERTIA = "1e-5"

EXPECTED_REVOLUTE_ORDER = [
    "volcaniarm_left_elbow_joint",
    "volcaniarm_left_arm_joint",
    "volcaniarm_right_elbow_joint",
    "volcaniarm_right_arm_joint",
]
EXPECTED_LINKS = {
    "base_link",
    "fl_table_leg_link", "fl_caster_wheel_mount_link", "fl_caster_wheel_link",
    "fr_table_leg_link", "fr_caster_wheel_mount_link", "fr_caster_wheel_link",
    "rl_table_leg_link", "rl_caster_wheel_mount_link", "rl_caster_wheel_link",
    "rr_table_leg_link", "rr_caster_wheel_mount_link", "rr_caster_wheel_link",
    "volcaniarm_base_link",
    "volcaniarm_left_elbow_link", "volcaniarm_left_arm_link",
    "volcaniarm_right_elbow_link", "volcaniarm_right_arm_link",
    "camera_mount_linear_link", "camera_mount_rev_link",
    "camera_link", "camera_link_optical",
    "left_ee_link", "right_ee_link",
}


def die(msg: str) -> None:
    print(f"[sync_urdf] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run_xacro(ws: Path) -> str:
    xacro_file = ws / XACRO_REL
    if not xacro_file.exists():
        die(f"no xacro at {xacro_file} — wrong --ws?")
    setup = ws / "install/setup.bash"
    if not setup.exists():
        die(f"{setup} missing — build the workspace first (colcon build)")
    cmd = f"source {setup} && xacro {xacro_file} " + " ".join(XACRO_ARGS)
    res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        die(f"xacro failed:\n{res.stderr}")
    return res.stdout


def strip_blocks(root: ET.Element) -> None:
    for tag in STRIP_BLOCKS:
        for el in root.findall(tag):
            root.remove(el)


def remove_link(root: ET.Element, name: str) -> None:
    """Remove a link and every joint touching it (either side).

    Parent-side matters: stripping `world` must also take `world_to_base`
    (whose *parent* is world) with it — a joint referencing a nonexistent
    link crashes the Isaac URDF parser with a bare C++ `map::at`.
    """
    for link in root.findall("link"):
        if link.get("name") == name:
            root.remove(link)
    for joint in root.findall("joint"):
        for side in ("child", "parent"):
            el = joint.find(side)
            if el is not None and el.get("link") == name:
                root.remove(joint)
                break


def joint_by_name(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            return joint
    die(f"joint {name!r} not found in ws URDF")


def capture_closure(root: ET.Element) -> dict:
    """Read the closure-point geometry before the debug links are removed."""
    tip = joint_by_name(root, "right_arm_tip_joint").find("origin")
    closure = joint_by_name(root, "closure_joint").find("origin")
    tip_xyz = tip.get("xyz")
    closure_xyz = closure.get("xyz")
    closure_rpy_x = float(closure.get("rpy").split()[0])
    if tip_xyz != closure_xyz:
        die(f"closure point mismatch: right_arm_tip_joint xyz={tip_xyz} vs "
            f"closure_joint xyz={closure_xyz}")
    return {"xyz": tip_xyz, "rpy_x": closure_rpy_x}


def cross_check_close_loop(rpy_x: float) -> None:
    m = re.search(r"^CLOSURE_RPY_X\s*=\s*([0-9.]+)", CLOSE_LOOP.read_text(), re.M)
    if not m:
        die(f"CLOSURE_RPY_X not found in {CLOSE_LOOP}")
    if abs(float(m.group(1)) - rpy_x) > 1e-9:
        die(f"closure rpy-x drift: ws xacro has {rpy_x}, close_loop.py has "
            f"{m.group(1)} — update one of them first")


def make_ee(side: str, xyz: str) -> tuple[ET.Element, ET.Element]:
    link = ET.fromstring(
        f'<link name="{side}_ee_link">'
        f'<inertial><origin xyz="0 0 0" rpy="0 0 0" />'
        f'<mass value="{EE_MASS}" />'
        f'<inertia ixx="{EE_INERTIA}" ixy="0" ixz="0" iyy="{EE_INERTIA}" '
        f'iyz="0" izz="{EE_INERTIA}" /></inertial></link>'
    )
    joint = ET.fromstring(
        f'<joint name="{side}_ee_joint" type="fixed">'
        f'<origin xyz="{xyz}" rpy="0 0 0" />'
        f'<parent link="volcaniarm_{side}_arm_link" />'
        f'<child link="{side}_ee_link" /></joint>'
    )
    return link, joint


def add_ee_links(root: ET.Element, geom: dict) -> None:
    """Insert ee link+joint right after each arm's joint, mirroring the
    document order of the historical hand-maintained URDF."""
    children = list(root)
    for side in ("left", "right"):
        anchor = joint_by_name(root, f"volcaniarm_{side}_arm_joint")
        idx = children.index(anchor)
        link, joint = make_ee(side, geom["xyz"])
        root.insert(idx + 1, joint)
        root.insert(idx + 1, link)
        children = list(root)


def rewrite_mesh_paths(root: ET.Element) -> None:
    prefix = "package://volcaniarm_description/meshes/"
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename", "")
        if uri.startswith(prefix):
            mesh.set("filename", str(MESH_DIR / uri[len(prefix):]))
        elif uri.startswith("package://"):
            die(f"unexpected package uri {uri!r}")


def sync_meshes(ws: Path) -> None:
    src_dir = ws / MESH_SRC_REL
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    src = {p.name: p for p in src_dir.iterdir() if p.suffix.lower() == ".stl"}
    dst = {p.name: p for p in MESH_DIR.iterdir() if p.suffix.lower() == ".stl"}

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    for name, p in sorted(src.items()):
        if name not in dst or sha(dst[name]) != sha(p):
            (MESH_DIR / name).write_bytes(p.read_bytes())
            print(f"[sync_urdf] mesh updated: {name}")
    for name in sorted(set(dst) - set(src)):
        dst[name].unlink()
        print(f"[sync_urdf] mesh removed (orphan): {name}")


def verify(root: ET.Element) -> None:
    links = {l.get("name") for l in root.findall("link")}
    if links != EXPECTED_LINKS:
        extra, missing = links - EXPECTED_LINKS, EXPECTED_LINKS - links
        die(f"link set mismatch — extra={sorted(extra)} missing={sorted(missing)}")
    revolute = [j.get("name") for j in root.findall("joint")
                if j.get("type") == "revolute"]
    if revolute != EXPECTED_REVOLUTE_ORDER:
        die(f"revolute joint order {revolute} != contract {EXPECTED_REVOLUTE_ORDER}")
    for jname in ("camera_mount_linear_joint", "camera_mount_rev_joint"):
        if joint_by_name(root, jname).get("type") != "fixed":
            die(f"{jname} must be fixed in the ws xacro")
    # Every joint must reference existing links — a dangling reference
    # crashes the Isaac URDF parser with an opaque C++ map::at.
    for joint in root.findall("joint"):
        for side in ("parent", "child"):
            ref = joint.find(side).get("link")
            if ref not in links:
                die(f"joint {joint.get('name')!r} references missing link {ref!r}")
    for mesh in root.iter("mesh"):
        if not Path(mesh.get("filename")).exists():
            die(f"mesh missing on disk: {mesh.get('filename')}")


def render(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    header = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<!-- GENERATED by scripts/sync_urdf.py from volcaniarm_ws's\n"
        "     volcaniarm.urdf.xacro - do not hand-edit. Edit the ws xacro and\n"
        "     re-run:  python3 scripts/sync_urdf.py  -->\n"
    )
    return header + body + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ws", type=Path,
                    default=Path("~/workspaces/volcaniarm_ws").expanduser())
    ap.add_argument("--check", action="store_true",
                    help="diff-only: exit 1 if the committed URDF is stale")
    ap.add_argument("--skip-meshes", action="store_true")
    args = ap.parse_args()

    root = ET.fromstring(run_xacro(args.ws))
    strip_blocks(root)
    geom = capture_closure(root)
    cross_check_close_loop(geom["rpy_x"])
    for name in STRIP_LINKS + DEBUG_LINKS:
        remove_link(root, name)
    add_ee_links(root, geom)
    rewrite_mesh_paths(root)
    if not args.skip_meshes and not args.check:
        sync_meshes(args.ws)
    verify(root)
    out = render(root)

    if args.check:
        current = OUT_URDF.read_text() if OUT_URDF.exists() else ""
        if current != out:
            die("committed URDF is stale — re-run scripts/sync_urdf.py")
        print("[sync_urdf] up to date")
        return

    OUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_URDF.write_text(out)
    print(f"[sync_urdf] wrote {OUT_URDF} "
          f"({len(root.findall('link'))} links, "
          f"{len(root.findall('joint'))} joints, closure at {geom['xyz']})")


if __name__ == "__main__":
    main()
