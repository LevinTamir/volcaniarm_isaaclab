"""Author the loop-closure overlay for the volcaniarm 5-bar.

Reads:   assets/usd/volcaniarm.usd         (open-tree base authored by convert_urdf.py)
Writes:  assets/usd/volcaniarm_closed.usd  (sublayers base + adds `closure_joint`)

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/close_loop.py

The closure joint:
  * name:    `closure_joint` (matches volcaniarm_controller passive_joints list)
  * type:    UsdPhysics.RevoluteJoint about the X axis
  * bodies:  left_ee_link <-> right_ee_link
  * frame offset on body1: rpy X = 1.8398 rad (per volcaniarm_controllers.yaml)
  * excluded from the articulation so PhysX treats it as a loop constraint
"""

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

from math import cos, sin
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

PROJECT = Path(__file__).resolve().parent.parent
BASE_USD = PROJECT / "assets/usd/volcaniarm.usd"
OVERLAY_USD = PROJECT / "assets/usd/volcaniarm_closed.usd"
CLOSURE_RPY_X = 1.8398


def main() -> None:
    base_stage = Usd.Stage.Open(str(BASE_USD))
    if base_stage is None:
        raise FileNotFoundError(f"Run convert_urdf.py first — missing {BASE_USD}")

    root_prim = base_stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError("Base USD has no default prim.")
    root_path = root_prim.GetPath()
    left_ee = root_path.AppendChild("left_ee_link")
    right_ee = root_path.AppendChild("right_ee_link")

    for path in (left_ee, right_ee):
        if not base_stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Base USD missing expected prim: {path}")

    if OVERLAY_USD.exists():
        OVERLAY_USD.unlink()
    overlay = Usd.Stage.CreateNew(str(OVERLAY_USD))
    # Pixar USD's `Usd.Stage.CreateNew` defaults to Y-up with mpu=0.01 — both
    # wrong for this content (authored Z-up in meters). Force-set so anyone
    # who opens this USD directly in Kit sees it upright at real scale.
    UsdGeom.SetStageUpAxis(overlay, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(overlay, 1.0)
    overlay.GetRootLayer().subLayerPaths.append(f"./{BASE_USD.name}")

    overlay_root = overlay.OverridePrim(root_path)
    overlay.SetDefaultPrim(overlay_root)

    joint_path = root_path.AppendChild("closure_joint")
    joint = UsdPhysics.RevoluteJoint.Define(overlay, joint_path)

    joint.CreateBody0Rel().SetTargets([left_ee])
    joint.CreateBody1Rel().SetTargets([right_ee])
    joint.CreateAxisAttr("X")

    half = CLOSURE_RPY_X / 2.0
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(cos(half), sin(half), 0.0, 0.0))

    joint.GetPrim().CreateAttribute(
        "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool
    ).Set(True)

    overlay.Save()
    print(f"Wrote: {OVERLAY_USD}")


if __name__ == "__main__":
    main()
    _app.close()
