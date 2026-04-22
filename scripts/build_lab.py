"""Author the lab-room visual overlay for the volcaniarm ROS2 scene.

Reads:   assets/usd/volcaniarm_ros2.usd   (robot + ground + camera + ROS2 graph)
Writes:  assets/usd/volcaniarm_lab.usd    (sublayers _ros2 + adds workshop decor)

Pure visual overlay: no physics APIs, no graph edits, no drive changes. Open
this USD in Isaac Sim GUI when you want the ROS2 demo to look like a
workshop — concrete floor, blue pegboard workbench, side desk, stool, and
a small potted plant matching the Gazebo `lab.sdf` pose so the two sims
share a visual reference.

The existing /World/GroundPlane cube from _ros2 is left alone (PhysX needs
its collision). The lab floor sits 1 cm above it; inside the 8x8 m room
the lab floor is what's visible; outside the walls, the grey ground shows.

A RectLight sits under the ceiling because the enclosed walls block the
outer DomeLight — without it the interior would render black.

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/build_lab.py
"""

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

PROJECT = Path(__file__).resolve().parent.parent
BASE_USD = PROJECT / "assets/usd/volcaniarm_ros2.usd"
OVERLAY_USD = PROJECT / "assets/usd/volcaniarm_lab.usd"

LAB_PATH = "/World/Lab"
FLOOR_Z_TOP = 0.001          # 1 mm above grey-ground top (z=0) — just enough
                             # to dodge z-fight without a visible step under the wheels
ROOM_HALF = 4.0              # interior half-width → 8x8 m room
WALL_THICKNESS = 0.1
WALL_HEIGHT = 3.5
CEILING_THICKNESS = 0.1
FLOOR_THICKNESS = 0.02

# Colors picked to match the photo of the real lab
FLOOR_COLOR = Gf.Vec3f(0.72, 0.70, 0.68)     # concrete light grey
WALL_COLOR = Gf.Vec3f(0.92, 0.91, 0.88)      # off-white
CEILING_COLOR = Gf.Vec3f(0.85, 0.85, 0.85)
PEGBOARD_COLOR = Gf.Vec3f(0.18, 0.42, 0.68)  # workshop blue
BENCH_TOP_COLOR = Gf.Vec3f(0.45, 0.32, 0.20) # brown wood
BENCH_LEG_COLOR = Gf.Vec3f(0.30, 0.30, 0.32) # dark grey metal
DESK_TOP_COLOR = Gf.Vec3f(0.55, 0.40, 0.25)  # lighter wood
POT_COLOR = Gf.Vec3f(0.55, 0.30, 0.18)       # terracotta
POT_RIM_COLOR = Gf.Vec3f(0.60, 0.35, 0.22)   # slightly lighter rim

# NVIDIA Omniverse sample-asset S3 root (Isaac Sim 5.1 bundle). Kit resolves
# http(s) asset URLs and caches them locally on first load, so referencing
# by URL keeps the scene self-contained in git while giving us real meshes.
NVIDIA_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
PLANT_USD = f"{NVIDIA_ASSET_ROOT}/NVIDIA/Assets/Vegetation/Plant_Tropical/Japanese_Painted_Fern.usd"

LIGHT_COLOR = Gf.Vec3f(1.00, 0.98, 0.95)
LIGHT_INTENSITY = 5000.0

# Default viewport pose — front-right of the arm, slightly above, aimed
# at the end-effector height. Echoes the angle in the reference photo.
# Base_link now lives at z=0.98, so the arm's working volume is roughly
# z in [0.98, 1.5]; target and camera pulled up to match.
VIEW_POS = Gf.Vec3d(2.0, -2.5, 1.5)
VIEW_TARGET = Gf.Vec3d(0.0, 0.0, 1.1)


def _add_box(stage, path, scale, translate, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetScale(Gf.Vec3f(*scale))
    cube.CreateDisplayColorAttr(Vt.Vec3fArray([color]))
    return cube


def _add_cylinder(stage, path, radius, height, translate, color):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(*translate))
    cyl.CreateDisplayColorAttr(Vt.Vec3fArray([color]))
    return cyl


def _add_sphere(stage, path, radius, translate, color, scale=(1.0, 1.0, 1.0)):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    xform = UsdGeom.XformCommonAPI(sph)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetScale(Gf.Vec3f(*scale))
    sph.CreateDisplayColorAttr(Vt.Vec3fArray([color]))
    return sph


def _add_room_shell(overlay):
    wall_span = 2 * ROOM_HALF + 2 * WALL_THICKNESS
    wall_center_z = FLOOR_Z_TOP + WALL_HEIGHT / 2.0
    wall_offset = ROOM_HALF + WALL_THICKNESS / 2.0

    _add_box(
        overlay, f"{LAB_PATH}/Floor",
        scale=(wall_span, wall_span, FLOOR_THICKNESS),
        translate=(0.0, 0.0, FLOOR_Z_TOP - FLOOR_THICKNESS / 2.0),
        color=FLOOR_COLOR,
    )

    for name, sx, sy, tx, ty in [
        ("WallNorth", wall_span, WALL_THICKNESS, 0.0, wall_offset),
        ("WallSouth", wall_span, WALL_THICKNESS, 0.0, -wall_offset),
        ("WallEast", WALL_THICKNESS, wall_span, wall_offset, 0.0),
        ("WallWest", WALL_THICKNESS, wall_span, -wall_offset, 0.0),
    ]:
        _add_box(
            overlay, f"{LAB_PATH}/{name}",
            scale=(sx, sy, WALL_HEIGHT),
            translate=(tx, ty, wall_center_z),
            color=WALL_COLOR,
        )

    ceiling_center_z = FLOOR_Z_TOP + WALL_HEIGHT + CEILING_THICKNESS / 2.0
    _add_box(
        overlay, f"{LAB_PATH}/Ceiling",
        scale=(wall_span, wall_span, CEILING_THICKNESS),
        translate=(0.0, 0.0, ceiling_center_z),
        color=CEILING_COLOR,
    )
    return ceiling_center_z


def _add_workbench(overlay):
    # West-wall workbench — behind the robot (robot faces +X). 3 m long along
    # the wall (Y axis), 0.7 m deep out from the wall (X axis), top 0.9 m up.
    # Blue pegboard panel mounted flush to the wall behind it.
    bench_l, bench_d, bench_th = 3.0, 0.7, 0.05  # length-along-wall, depth, top thickness
    top_z = FLOOR_Z_TOP + 0.90
    x_center = -(ROOM_HALF - bench_d / 2.0 - 0.02)  # 2 cm clear of wall

    _add_box(
        overlay, f"{LAB_PATH}/BenchTop",
        scale=(bench_d, bench_l, bench_th),
        translate=(x_center, 0.0, top_z - bench_th / 2.0),
        color=BENCH_TOP_COLOR,
    )

    leg_h = top_z - bench_th - FLOOR_Z_TOP
    leg_size = 0.05
    leg_z = FLOOR_Z_TOP + leg_h / 2.0
    leg_y = bench_l / 2.0 - leg_size
    leg_x_front = x_center - bench_d / 2.0 + leg_size / 2.0
    leg_x_back = x_center + bench_d / 2.0 - leg_size / 2.0
    for i, (lx, ly) in enumerate([
        (leg_x_front, leg_y), (leg_x_front, -leg_y),
        (leg_x_back, leg_y), (leg_x_back, -leg_y),
    ]):
        _add_box(
            overlay, f"{LAB_PATH}/BenchLeg{i}",
            scale=(leg_size, leg_size, leg_h),
            translate=(lx, ly, leg_z),
            color=BENCH_LEG_COLOR,
        )

    peg_l, peg_h, peg_thickness = bench_l, 1.2, 0.03
    peg_bottom = top_z + 0.02
    peg_center_z = peg_bottom + peg_h / 2.0
    peg_x = -(ROOM_HALF - peg_thickness / 2.0 - 0.001)  # flush to wall, no z-fight
    _add_box(
        overlay, f"{LAB_PATH}/Pegboard",
        scale=(peg_thickness, peg_l, peg_h),
        translate=(peg_x, 0.0, peg_center_z),
        color=PEGBOARD_COLOR,
    )


def _add_desk(overlay):
    # North-wall desk — side wall relative to the robot's forward axis.
    # 1.4 m long along X, 0.6 m deep out from the wall.
    desk_w, desk_d, desk_th = 1.4, 0.6, 0.05
    top_z = FLOOR_Z_TOP + 0.75
    y_center = ROOM_HALF - desk_d / 2.0 - 0.02
    x_center = 1.5  # offset from centerline so it sits clear of the arm sweep

    _add_box(
        overlay, f"{LAB_PATH}/DeskTop",
        scale=(desk_w, desk_d, desk_th),
        translate=(x_center, y_center, top_z - desk_th / 2.0),
        color=DESK_TOP_COLOR,
    )

    leg_h = top_z - desk_th - FLOOR_Z_TOP
    leg_size = 0.04
    leg_z = FLOOR_Z_TOP + leg_h / 2.0
    ly_front = y_center - desk_d / 2.0 + leg_size / 2.0
    ly_back = y_center + desk_d / 2.0 - leg_size / 2.0
    lx_plus = x_center + desk_w / 2.0 - leg_size
    lx_minus = x_center - desk_w / 2.0 + leg_size
    for i, (lx, ly) in enumerate([
        (lx_plus, ly_front), (lx_plus, ly_back),
        (lx_minus, ly_front), (lx_minus, ly_back),
    ]):
        _add_box(
            overlay, f"{LAB_PATH}/DeskLeg{i}",
            scale=(leg_size, leg_size, leg_h),
            translate=(lx, ly, leg_z),
            color=BENCH_LEG_COLOR,
        )


def _add_potted_plant(overlay):
    # XY pose matches Gazebo lab.sdf weed_target (-0.12, 0.25) — the two sims
    # share a visual reference. Terracotta pot as primitives; foliage is a
    # reference to NVIDIA's Japanese Painted Fern, drastically shrunk.
    x, y = -0.02, 0.25     # shifted +0.10 m in X from Gazebo weed_target pose
    # 40% smaller than the previous (0.08 / 0.12 / 0.088 / 0.012) sizing
    pot_r_outer, pot_h = 0.048, 0.072
    rim_r, rim_h = 0.053, 0.007

    _add_cylinder(
        overlay, f"{LAB_PATH}/PlantPot",
        radius=pot_r_outer, height=pot_h,
        translate=(x, y, FLOOR_Z_TOP + pot_h / 2.0),
        color=POT_COLOR,
    )
    _add_cylinder(
        overlay, f"{LAB_PATH}/PlantPotRim",
        radius=rim_r, height=rim_h,
        translate=(x, y, FLOOR_Z_TOP + pot_h - rim_h / 2.0),
        color=POT_RIM_COLOR,
    )

    # Japanese Painted Fern native bbox measured empirically: ~27 m tall,
    # origin is ~2.29 m above the geometry's bottom (asset is authored for
    # outdoor landscape use). Scale to a 35 cm tabletop plant and offset
    # the translate so the scaled asset's lowest point sits just below the
    # pot rim (a small bury reads as "stems in soil").
    PLANT_TARGET_H = 0.21       # 40% smaller than the previous 0.35 m
    PLANT_NATIVE_H = 27.2
    PLANT_NATIVE_MIN_Z = -2.29
    plant_scale = PLANT_TARGET_H / PLANT_NATIVE_H          # ~0.0077
    bury = 0.012
    pot_top_z = FLOOR_Z_TOP + pot_h
    plant_z = pot_top_z - plant_scale * PLANT_NATIVE_MIN_Z - bury

    # Reference + scale via named ops (not XformCommonAPI — the referenced
    # asset has xformOp:orient which breaks XformCommonAPI's expected schema).
    foliage = overlay.DefinePrim(f"{LAB_PATH}/PlantFoliage", "Xform")
    foliage.GetReferences().AddReference(PLANT_USD)
    foliage_xf = UsdGeom.Xformable(foliage)
    foliage_xf.AddTranslateOp(opSuffix="place").Set(Gf.Vec3d(x, y, plant_z))
    foliage_xf.AddScaleOp(opSuffix="shrink").Set(
        Gf.Vec3f(plant_scale, plant_scale, plant_scale)
    )


def _add_ceiling_light(overlay, ceiling_center_z):
    light_path = f"{LAB_PATH}/CeilingLight"
    light = UsdLux.RectLight.Define(overlay, light_path)
    light.CreateWidthAttr(4.0)
    light.CreateHeightAttr(4.0)
    light.CreateIntensityAttr(LIGHT_INTENSITY)
    light.CreateColorAttr(LIGHT_COLOR)
    UsdGeom.XformCommonAPI(light).SetTranslate(
        Gf.Vec3d(0.0, 0.0, ceiling_center_z - CEILING_THICKNESS / 2.0 - 0.01)
    )


def _add_named_view_camera(overlay):
    # Scene camera at the same pose as the Perspective viewport seed, so
    # there's a named camera in the dropdown to snap back to.
    cam = UsdGeom.Camera.Define(overlay, f"{LAB_PATH}/LabView")
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))

    view = Gf.Matrix4d()
    view.SetLookAt(VIEW_POS, VIEW_TARGET, Gf.Vec3d(0.0, 0.0, 1.0))
    cam.MakeMatrixXform().Set(view.GetInverse())


def _seed_viewport_pose(overlay):
    # Omniverse/Kit honors `customLayerData.cameraSettings` to seed the
    # viewport's default Perspective camera when a stage is first opened.
    # Without this the stock Kit start pose is far out and off-axis for
    # this scene.
    overlay.GetRootLayer().customLayerData = {
        "cameraSettings": {
            "Perspective": {
                "position": VIEW_POS,
                "target": VIEW_TARGET,
            },
            "boundCamera": "/OmniverseKit_Persp",
        }
    }


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
    # Pixar USD's `Usd.Stage.CreateNew` defaults to Y-up with mpu=0.01 — both
    # wrong for this content (authored Z-up in meters). Without these lines
    # the lab opens rotated and scaled 100x when sublayered onto Z-up _ros2.
    UsdGeom.SetStageUpAxis(overlay, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(overlay, 1.0)
    overlay.GetRootLayer().subLayerPaths.append(f"./{BASE_USD.name}")

    overlay_root = overlay.OverridePrim(root_path)
    overlay.SetDefaultPrim(overlay_root)

    UsdGeom.Xform.Define(overlay, LAB_PATH)

    ceiling_center_z = _add_room_shell(overlay)
    _add_workbench(overlay)
    _add_desk(overlay)
    _add_potted_plant(overlay)
    _add_ceiling_light(overlay, ceiling_center_z)
    _add_named_view_camera(overlay)
    _seed_viewport_pose(overlay)

    overlay.Save()
    print(f"Wrote: {OVERLAY_USD}")


if __name__ == "__main__":
    main()
    _app.close()
