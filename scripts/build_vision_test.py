"""Author the vision-test overlay for the volcaniarm ROS2 scene.

Reads:   assets/usd/volcaniarm_ros2.usd       (robot + ground + camera + ROS2 graph)
Writes:  assets/usd/volcaniarm_vision_test.usd

Adds a single green cylinder under the arm at the same pose used by
the Gazebo `vision_test.sdf` world, so the deployed RL vision
controller sees the same target across both simulators. Geometry
mirrors the training plant surrogate
(`reach_vision_env_cfg.py:plant`): radius 0.025 m, height 0.10 m,
diffuse colour (0.15, 0.55, 0.15).

Pure visual overlay: no physics APIs, no graph edits. Open in Isaac
Sim with the ROS2 bridge running to drive `volcaniarm_rl_vision_controller`.

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/build_vision_test.py
"""

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, Vt

PROJECT = Path(__file__).resolve().parent.parent
BASE_USD = PROJECT / "assets/usd/volcaniarm_ros2.usd"
OVERLAY_USD = PROJECT / "assets/usd/volcaniarm_vision_test.usd"

ROOT_PATH = "/World/VisionTest"
CYLINDER_PATH = f"{ROOT_PATH}/GreenCylinder"

# World-frame placement. Mirrors `vision_test.sdf` so a policy that
# reaches the cylinder in Gazebo also reaches it here.
#   X  -0.05 — moved forward (+X) from the original -0.15 so the
#                cylinder lands under the arm's reachable region in
#                IsaacSim's referenced robot pose. Tune empirically
#                with scripts/render_check.py if it needs more bias.
#   Y   0.25 — inside the ±0.5 workspace, offset for visibility
#   Z   0.05 — cylinder centre 5 cm above the ground; with height 0.10
#              the bottom sits flush at z=0
CYL_TRANSLATE = Gf.Vec3d(-0.05, 0.25, 0.05)
CYL_RADIUS = 0.025
CYL_HEIGHT = 0.10
CYL_COLOR = Gf.Vec3f(0.15, 0.55, 0.15)


def _add_green_cylinder(stage):
    cyl = UsdGeom.Cylinder.Define(stage, CYLINDER_PATH)
    cyl.CreateRadiusAttr(CYL_RADIUS)
    cyl.CreateHeightAttr(CYL_HEIGHT)
    cyl.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(cyl).SetTranslate(CYL_TRANSLATE)
    cyl.CreateDisplayColorAttr(Vt.Vec3fArray([CYL_COLOR]))
    return cyl


def main() -> None:
    base_stage = Usd.Stage.Open(str(BASE_USD))
    if base_stage is None:
        raise FileNotFoundError(f"Run add_ros2_graph.py first — missing {BASE_USD}")
    root_prim = base_stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError("Base USD has no default prim.")
    root_path = root_prim.GetPath()  # /World

    if OVERLAY_USD.exists():
        OVERLAY_USD.unlink()
    overlay = Usd.Stage.CreateNew(str(OVERLAY_USD))
    # Default Pixar Stage is Y-up + cm. Z-up + meters matches the rest
    # of the assets (see build_lab.py for the same fix).
    UsdGeom.SetStageUpAxis(overlay, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(overlay, 1.0)
    overlay.GetRootLayer().subLayerPaths.append(f"./{BASE_USD.name}")

    overlay_root = overlay.OverridePrim(root_path)
    overlay.SetDefaultPrim(overlay_root)

    UsdGeom.Xform.Define(overlay, ROOT_PATH)
    _add_green_cylinder(overlay)

    overlay.Save()
    print(f"Wrote: {OVERLAY_USD}")


if __name__ == "__main__":
    main()
    _app.close()
