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
FLOOR_Z_TOP = -0.97          # ~1 cm above grey-ground top (-0.98), dodges z-fight
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
STOOL_COLOR = Gf.Vec3f(0.15, 0.15, 0.15)     # black
STOOL_SEAT_COLOR = Gf.Vec3f(0.85, 0.75, 0.15)  # yellow
VASE_COLOR = Gf.Vec3f(0.08, 0.08, 0.08)      # black
PLANT_COLOR = Gf.Vec3f(0.12, 0.55, 0.15)     # leaf green

LIGHT_COLOR = Gf.Vec3f(1.00, 0.98, 0.95)
LIGHT_INTENSITY = 5000.0


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
    # North-wall workbench: 3 m wide, 0.7 m deep, top at 0.9 m above floor.
    # Pegboard panel behind rises 1.2 m up the wall.
    bench_w, bench_d, bench_h = 3.0, 0.7, 0.05
    top_z = FLOOR_Z_TOP + 0.90
    y_center = ROOM_HALF - bench_d / 2.0 - 0.02  # 2 cm clear of wall

    _add_box(
        overlay, f"{LAB_PATH}/BenchTop",
        scale=(bench_w, bench_d, bench_h),
        translate=(0.0, y_center, top_z - bench_h / 2.0),
        color=BENCH_TOP_COLOR,
    )

    leg_h = top_z - bench_h - FLOOR_Z_TOP
    leg_size = 0.05
    leg_z = FLOOR_Z_TOP + leg_h / 2.0
    leg_x = bench_w / 2.0 - leg_size
    leg_y_front = y_center - bench_d / 2.0 + leg_size / 2.0
    leg_y_back = y_center + bench_d / 2.0 - leg_size / 2.0
    for i, (lx, ly) in enumerate([
        (leg_x, leg_y_front), (-leg_x, leg_y_front),
        (leg_x, leg_y_back), (-leg_x, leg_y_back),
    ]):
        _add_box(
            overlay, f"{LAB_PATH}/BenchLeg{i}",
            scale=(leg_size, leg_size, leg_h),
            translate=(lx, ly, leg_z),
            color=BENCH_LEG_COLOR,
        )

    peg_w, peg_h, peg_thickness = bench_w, 1.2, 0.03
    peg_bottom = top_z + 0.02
    peg_center_z = peg_bottom + peg_h / 2.0
    peg_y = ROOM_HALF - peg_thickness / 2.0 - 0.001  # flush to wall, no z-fight
    _add_box(
        overlay, f"{LAB_PATH}/Pegboard",
        scale=(peg_w, peg_thickness, peg_h),
        translate=(0.0, peg_y, peg_center_z),
        color=PEGBOARD_COLOR,
    )


def _add_desk(overlay):
    # Smaller desk along the east wall (camera sits to the right of robot).
    desk_w, desk_d, desk_h = 1.4, 0.6, 0.05
    top_z = FLOOR_Z_TOP + 0.75
    x_center = ROOM_HALF - desk_d / 2.0 - 0.02
    y_center = -1.5  # offset from centerline so it's visible past the robot

    _add_box(
        overlay, f"{LAB_PATH}/DeskTop",
        scale=(desk_d, desk_w, desk_h),
        translate=(x_center, y_center, top_z - desk_h / 2.0),
        color=DESK_TOP_COLOR,
    )

    leg_h = top_z - desk_h - FLOOR_Z_TOP
    leg_size = 0.04
    leg_z = FLOOR_Z_TOP + leg_h / 2.0
    lx_back = x_center + desk_d / 2.0 - leg_size / 2.0
    lx_front = x_center - desk_d / 2.0 + leg_size / 2.0
    ly_plus = y_center + desk_w / 2.0 - leg_size
    ly_minus = y_center - desk_w / 2.0 + leg_size
    for i, (lx, ly) in enumerate([
        (lx_back, ly_plus), (lx_back, ly_minus),
        (lx_front, ly_plus), (lx_front, ly_minus),
    ]):
        _add_box(
            overlay, f"{LAB_PATH}/DeskLeg{i}",
            scale=(leg_size, leg_size, leg_h),
            translate=(lx, ly, leg_z),
            color=BENCH_LEG_COLOR,
        )


def _add_stool(overlay):
    # Wheeled workshop stool off to the west, clear of the robot's swept area.
    base_r, base_h = 0.25, 0.03
    post_r, post_h = 0.03, 0.40
    seat_r, seat_h = 0.17, 0.05
    x, y = -2.4, -1.6

    _add_cylinder(
        overlay, f"{LAB_PATH}/StoolBase",
        radius=base_r, height=base_h,
        translate=(x, y, FLOOR_Z_TOP + base_h / 2.0),
        color=STOOL_COLOR,
    )
    _add_cylinder(
        overlay, f"{LAB_PATH}/StoolPost",
        radius=post_r, height=post_h,
        translate=(x, y, FLOOR_Z_TOP + base_h + post_h / 2.0),
        color=STOOL_COLOR,
    )
    _add_cylinder(
        overlay, f"{LAB_PATH}/StoolSeat",
        radius=seat_r, height=seat_h,
        translate=(x, y, FLOOR_Z_TOP + base_h + post_h + seat_h / 2.0),
        color=STOOL_SEAT_COLOR,
    )


def _add_potted_plant(overlay):
    # Pose matches Gazebo lab.sdf weed_target: (-0.12, 0.25, 0) in world where
    # the Gazebo floor is z=0. Translate into Isaac by setting base at
    # FLOOR_Z_TOP. Dimensions copied from lab.sdf: vase radius 0.04, height
    # 0.15. Plant foliage is a green sphere sitting on the vase top (the
    # Gazebo version loads a .dae mesh; primitives keep build_lab.py free of
    # asset-conversion plumbing and still read as a potted plant).
    x, y = -0.12, 0.25
    vase_r, vase_h = 0.04, 0.15
    foliage_r = 0.07

    _add_cylinder(
        overlay, f"{LAB_PATH}/PlantVase",
        radius=vase_r, height=vase_h,
        translate=(x, y, FLOOR_Z_TOP + vase_h / 2.0),
        color=VASE_COLOR,
    )
    _add_sphere(
        overlay, f"{LAB_PATH}/PlantFoliage",
        radius=foliage_r,
        translate=(x, y, FLOOR_Z_TOP + vase_h + foliage_r * 0.85),
        color=PLANT_COLOR,
        scale=(1.0, 1.0, 1.15),  # slightly taller than wide — bush shape
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
    overlay.GetRootLayer().subLayerPaths.append(f"./{BASE_USD.name}")

    overlay_root = overlay.OverridePrim(root_path)
    overlay.SetDefaultPrim(overlay_root)

    UsdGeom.Xform.Define(overlay, LAB_PATH)

    ceiling_center_z = _add_room_shell(overlay)
    _add_workbench(overlay)
    _add_desk(overlay)
    _add_stool(overlay)
    _add_potted_plant(overlay)
    _add_ceiling_light(overlay, ceiling_center_z)

    overlay.Save()
    print(f"Wrote: {OVERLAY_USD}")


if __name__ == "__main__":
    main()
    _app.close()
