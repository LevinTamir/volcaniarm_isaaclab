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

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, Vt

PROJECT = Path(__file__).resolve().parent.parent
BASE_USD = PROJECT / "assets/usd/volcaniarm_ros2.usd"
OVERLAY_USD = PROJECT / "assets/usd/volcaniarm_lab.usd"

LAB_PATH = "/World/Lab"
FLOOR_Z_TOP = 0.001          # 1 mm above grey-ground top (z=0) — just enough
                             # to dodge z-fight without a visible step under the wheels
# Room interior bounds, expressed as distances from the robot at the origin
# rather than a symmetric half-width — the rig sits close to the back wall in
# the real lab, and that wall is what fills the camera's background.
# Robot faces +X, so -X is "behind the arm".
ROOM_BACK_X = -0.5           # white wall 0.5 m behind the arm
ROOM_FRONT_X = 3.0
ROOM_HALF_Y = 2.0
WALL_THICKNESS = 0.1
WALL_HEIGHT = 2.6

# Open-topped, open-fronted box. The room exists to give the camera a
# realistic *background* (white wall behind the arm, side walls at the edges
# of frame) — it is not meant to be a sealed volume. A ceiling and a front
# wall only get in the way when you orbit the viewport out of the room, and
# neither is ever visible to the arm-mounted camera.
BUILD_CEILING = False
BUILD_FRONT_WALL = False      # the +X wall, in front of the arm

# Workshop furniture. Off by default: none of it is visible to the
# arm-mounted camera, and the vision policy only ever sees the green mask, so
# it contributes nothing to the env. Kept behind flags because it is useful
# for renders and for matching the Gazebo lab visually.
BUILD_WORKBENCH = False       # bench + blue pegboard
BUILD_DESK = False
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

# Black non-reflective rubber mat under the rig. The concrete floor is kept:
# in the real lab the mat covers only part of the view and light concrete is
# visible past its edge, which is what the photo shows.
MAT_COLOR = Gf.Vec3f(0.04, 0.04, 0.04)
MAT_ROUGHNESS = 0.95         # matte rubber — see _add_floor_mat
# Measured: 0.9 m deep (X, robot's forward axis) x 2.0 m wide (Y, the arm's
# sweep direction). Centred on the robot, so it stops just shy of the back
# wall at x=-0.5.
MAT_SIZE = (0.9, 2.0)
MAT_THICKNESS = 0.006
MAT_CENTER = (0.0, 0.0)

# The 3D-printed weed, built by scripts/convert_weed.py. Origin is at the
# canopy apex, so translate by +height to seat the base on the mat.
WEED_USD = PROJECT / "assets/usd/fake_weed.usd"
WEED_USD_HEIGHT = 0.115      # height baked into the asset by convert_weed.py
# Matches the training task (contract.WEED_HEIGHT_M) so the visual reference
# and the trained-against weed are the same object. Print the STL at ~1.57x
# its authored 73.35 mm to match.
WEED_HEIGHT = 0.115
WEED_COLOR = Gf.Vec3f(0.24, 0.75, 0.51)
WEED_XY = (0.071, 0.10)      # on the mat, inside the arm's reachable Y span

# NVIDIA Omniverse sample-asset S3 root (Isaac Sim 5.1 bundle). Kit resolves
# http(s) asset URLs and caches them locally on first load, so referencing
# by URL keeps the scene self-contained in git while giving us real meshes.
NVIDIA_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
PLANT_USD = f"{NVIDIA_ASSET_ROOT}/NVIDIA/Assets/Vegetation/Plant_Tropical/Japanese_Painted_Fern.usd"

LIGHT_COLOR = Gf.Vec3f(1.00, 0.98, 0.95)
LIGHT_INTENSITY = 5000.0

# Default viewport pose — front-right of the arm, above the mat, aimed at the
# middle of the working volume. Echoes the angle in the reference photo.
#
# The subject spans from the weed on the mat (z~0.07) up to base_link
# (z=0.98), so the target sits between them rather than at base height; aiming
# at 1.1 puts the mat and weed near the bottom edge of frame.
#
# Must stay INSIDE the room — see the assert in _seed_viewport_pose. The
# previous pose (y=-2.5) silently ended up outside the south wall when the
# room shrank from 8x8 to 3.5x4.0, which put the default view inside a wall.
VIEW_POS = Gf.Vec3d(1.6, -1.2, 1.1)
VIEW_TARGET = Gf.Vec3d(0.07, 0.0, 0.55)


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
    span_x = ROOM_FRONT_X - ROOM_BACK_X + 2 * WALL_THICKNESS
    span_y = 2 * ROOM_HALF_Y + 2 * WALL_THICKNESS
    mid_x = 0.5 * (ROOM_FRONT_X + ROOM_BACK_X)
    wall_center_z = FLOOR_Z_TOP + WALL_HEIGHT / 2.0

    _add_box(
        overlay, f"{LAB_PATH}/Floor",
        scale=(span_x, span_y, FLOOR_THICKNESS),
        translate=(mid_x, 0.0, FLOOR_Z_TOP - FLOOR_THICKNESS / 2.0),
        color=FLOOR_COLOR,
    )

    y_off = ROOM_HALF_Y + WALL_THICKNESS / 2.0
    walls = [
        ("WallNorth", span_x, WALL_THICKNESS, mid_x, y_off),
        ("WallSouth", span_x, WALL_THICKNESS, mid_x, -y_off),
        ("WallWest", WALL_THICKNESS, span_y, ROOM_BACK_X - WALL_THICKNESS / 2.0, 0.0),
    ]
    if BUILD_FRONT_WALL:
        walls.append(
            ("WallEast", WALL_THICKNESS, span_y, ROOM_FRONT_X + WALL_THICKNESS / 2.0, 0.0)
        )
    for name, sx, sy, tx, ty in walls:
        _add_box(
            overlay, f"{LAB_PATH}/{name}",
            scale=(sx, sy, WALL_HEIGHT),
            translate=(tx, ty, wall_center_z),
            color=WALL_COLOR,
        )

    ceiling_center_z = FLOOR_Z_TOP + WALL_HEIGHT + CEILING_THICKNESS / 2.0
    if BUILD_CEILING:
        _add_box(
            overlay, f"{LAB_PATH}/Ceiling",
            scale=(span_x, span_y, CEILING_THICKNESS),
            translate=(mid_x, 0.0, ceiling_center_z),
            color=CEILING_COLOR,
        )
    # Returned regardless: the ceiling light hangs at this height whether or
    # not the ceiling slab itself is built.
    return ceiling_center_z


def _add_workbench(overlay):
    """Workbench + blue pegboard, on the SOUTH side wall.

    Deliberately not behind the arm any more. The back wall is what fills the
    camera's background, and in the real lab that is bare white — a 3 m blue
    pegboard there is both wrong and a large saturated distractor sitting
    directly behind the target. Moved to the -Y side wall, out of frame.
    """
    bench_l, bench_d, bench_th = 2.0, 0.6, 0.05  # length-along-wall(X), depth(Y), top
    top_z = FLOOR_Z_TOP + 0.90
    y_center = -(ROOM_HALF_Y - bench_d / 2.0 - 0.02)  # 2 cm clear of wall
    x_center = 1.4  # forward of the arm, clear of its sweep

    _add_box(
        overlay, f"{LAB_PATH}/BenchTop",
        scale=(bench_l, bench_d, bench_th),
        translate=(x_center, y_center, top_z - bench_th / 2.0),
        color=BENCH_TOP_COLOR,
    )

    leg_h = top_z - bench_th - FLOOR_Z_TOP
    leg_size = 0.05
    leg_z = FLOOR_Z_TOP + leg_h / 2.0
    leg_x = bench_l / 2.0 - leg_size
    leg_y_front = y_center + bench_d / 2.0 - leg_size / 2.0
    leg_y_back = y_center - bench_d / 2.0 + leg_size / 2.0
    for i, (lx, ly) in enumerate([
        (x_center + leg_x, leg_y_front), (x_center - leg_x, leg_y_front),
        (x_center + leg_x, leg_y_back), (x_center - leg_x, leg_y_back),
    ]):
        _add_box(
            overlay, f"{LAB_PATH}/BenchLeg{i}",
            scale=(leg_size, leg_size, leg_h),
            translate=(lx, ly, leg_z),
            color=BENCH_LEG_COLOR,
        )

    peg_l, peg_h, peg_thickness = bench_l, 1.0, 0.03
    peg_center_z = top_z + 0.02 + peg_h / 2.0
    peg_y = -(ROOM_HALF_Y - peg_thickness / 2.0 - 0.001)  # flush to wall
    _add_box(
        overlay, f"{LAB_PATH}/Pegboard",
        scale=(peg_l, peg_thickness, peg_h),
        translate=(x_center, peg_y, peg_center_z),
        color=PEGBOARD_COLOR,
    )


def _add_desk(overlay):
    # North-wall desk — side wall relative to the robot's forward axis.
    # 1.4 m long along X, 0.6 m deep out from the wall.
    desk_w, desk_d, desk_th = 1.4, 0.6, 0.05
    top_z = FLOOR_Z_TOP + 0.75
    y_center = ROOM_HALF_Y - desk_d / 2.0 - 0.02
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


def _usd_height(usd_path):
    """Z extent of a USD asset's geometry, in metres.

    Opened on a throwaway stage so this never disturbs the overlay being
    authored.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"cannot open {usd_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    return float(rng.GetMax()[2] - rng.GetMin()[2])


def _add_floor_mat(overlay):
    """Black rubber mat laid on the concrete, directly under the rig.

    Sits 1 mm proud of the floor so there is no z-fighting between the two
    coplanar boxes.

    The material binding is the load-bearing part. `_add_box` only authors
    `displayColor`, which is a viewport *hint* — under RTX the prim falls back
    to the default surface, which is glossy, so a "black" mat still renders
    with bright specular highlights. Binding an explicit PreviewSurface at
    roughness ~0.95 with no specular is what actually makes it matte.
    """
    mx, my = MAT_CENTER
    mat = _add_box(
        overlay, f"{LAB_PATH}/FloorMat",
        scale=(MAT_SIZE[0], MAT_SIZE[1], MAT_THICKNESS),
        translate=(mx, my, FLOOR_Z_TOP + MAT_THICKNESS / 2.0),
        color=MAT_COLOR,
    )
    _bind_color(
        overlay, mat.GetPrim(), MAT_COLOR, f"{LAB_PATH}/Looks/MatBlack",
        roughness=MAT_ROUGHNESS, specular=0.0,
    )
    return FLOOR_Z_TOP + MAT_THICKNESS


def _add_weed(overlay, mat_top_z):
    """The 3D-printed weed standing on the mat.

    Replaces the decorative Japanese Painted Fern that used to stand in for a
    plant here. Two reasons: the real lab has no fern, and — more importantly
    — the vision policy segments on *green*, so a second green object in frame
    is exactly the failure mode the blob isolation exists to handle. Leaving a
    large decorative fern in the world would give the ROS2/IsaacSim deploy
    tests a competing blob that reality doesn't have.

    The USD origin is the canopy apex (see scripts/convert_weed.py), so
    translating to `mat_top_z + WEED_HEIGHT` puts the base on the mat.
    """
    if not WEED_USD.exists():
        raise FileNotFoundError(
            f"Run scripts/convert_weed.py first — missing {WEED_USD}"
        )
    x, y = WEED_XY
    # Measure the asset's real height instead of trusting a hand-copied
    # constant. These two drifted once already: build_lab baked
    # scale = 0.07/0.20 = 0.35, then convert_weed.py was re-run at 0.115 m, so
    # the lab world silently rendered a 4 cm weed floating 3 cm off the mat —
    # and the deployed policy saw a target ~8x too small in pixel area. Reading
    # the source of truth makes that impossible.
    asset_h = _usd_height(WEED_USD)
    if abs(asset_h - WEED_USD_HEIGHT) > 1e-4:
        print(
            f"[build_lab] note: {WEED_USD.name} is {asset_h:.4f} m tall, "
            f"WEED_USD_HEIGHT says {WEED_USD_HEIGHT:.4f} — using the measured value"
        )
    scale = WEED_HEIGHT / asset_h
    weed = overlay.DefinePrim(f"{LAB_PATH}/Weed", "Xform")
    weed.GetReferences().AddReference(str(WEED_USD))
    xf = UsdGeom.Xformable(weed)
    xf.AddTranslateOp(opSuffix="place").Set(
        Gf.Vec3d(x, y, mat_top_z + WEED_HEIGHT)
    )
    xf.AddScaleOp(opSuffix="size").Set(Gf.Vec3f(scale, scale, scale))

    # Bind the green material here rather than baking it into the asset, so
    # the colour stays in one place alongside the training-side WEED_COLOR.
    _bind_color(overlay, weed, WEED_COLOR, f"{LAB_PATH}/Looks/WeedGreen")


def _bind_color(overlay, prim, color, mat_path, roughness=0.6, specular=0.5):
    """Author a PreviewSurface material and bind it to `prim`."""
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(overlay, mat_path)
    shader = UsdShade.Shader.Define(overlay, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(specular)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
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


def _assert_view_inside_room():
    """Fail loudly if the seeded viewport pose is outside the walls.

    This is a guard against exactly the bug it was written for: VIEW_POS was
    authored for an 8x8 m room and silently ended up beyond the south wall
    when the room shrank, so every fresh open of the stage started inside
    geometry. A stale camera constant is invisible until someone opens the
    USD, which makes it worth checking at build time.
    """
    x, y, z = VIEW_POS[0], VIEW_POS[1], VIEW_POS[2]
    inside = (
        ROOM_BACK_X < x < ROOM_FRONT_X
        and -ROOM_HALF_Y < y < ROOM_HALF_Y
        and FLOOR_Z_TOP < z < FLOOR_Z_TOP + WALL_HEIGHT
    )
    if not inside:
        raise ValueError(
            f"VIEW_POS {tuple(VIEW_POS)} is outside the room "
            f"(x in ({ROOM_BACK_X}, {ROOM_FRONT_X}), "
            f"y in ({-ROOM_HALF_Y}, {ROOM_HALF_Y}), "
            f"z in ({FLOOR_Z_TOP}, {FLOOR_Z_TOP + WALL_HEIGHT})) — "
            "the default viewport would open inside a wall."
        )


def _seed_viewport_pose(overlay):
    # Omniverse/Kit honors `customLayerData.cameraSettings` to seed the
    # viewport's default Perspective camera when a stage is first opened.
    # Without this the stock Kit start pose is far out and off-axis for
    # this scene.
    _assert_view_inside_room()
    overlay.GetRootLayer().customLayerData = {
        "cameraSettings": {
            "Perspective": {
                "position": VIEW_POS,
                "target": VIEW_TARGET,
            },
            "boundCamera": "/OmniverseKit_Persp",
        }
    }


def _seat_robot_on_mat(overlay, base_stage, mat_top_z):
    """Raise the robot by the mat thickness so the wheels rest on the mat.

    _ros2 seats the wheel bottoms at world z=0 (its ground has no mat).
    Here the mat sits on the lab floor under the rig, so without this
    override the wheels would sink `mat_top_z` into it. The base stage
    already authors the `xformOp:translate:raise` op; the overlay's root
    layer is stronger than its sublayer, so re-authoring the same attr
    with the mat height added is enough — no new xform ops.
    """
    robot_path = "/World/volcaniarm"
    base_raise_attr = base_stage.GetPrimAtPath(robot_path).GetAttribute(
        "xformOp:translate:raise")
    if not base_raise_attr or base_raise_attr.Get() is None:
        raise RuntimeError(
            f"{robot_path} has no xformOp:translate:raise in {BASE_USD.name} — "
            "regenerate it with add_ros2_graph.py first")
    lift = Gf.Vec3d(base_raise_attr.Get()) + Gf.Vec3d(0.0, 0.0, mat_top_z)
    over = overlay.OverridePrim(robot_path)
    over.CreateAttribute(
        "xformOp:translate:raise", Sdf.ValueTypeNames.Double3).Set(lift)
    print(f"[build_lab] robot raise override: {lift[2]:.4f} m "
          f"(wheels on mat top at z={mat_top_z:.4f})")


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
    mat_top_z = _add_floor_mat(overlay)
    _seat_robot_on_mat(overlay, base_stage, mat_top_z)
    if BUILD_WORKBENCH:
        _add_workbench(overlay)
    if BUILD_DESK:
        _add_desk(overlay)
    # The decorative fern is deliberately not built any more — see _add_weed.
    _add_weed(overlay, mat_top_z)
    _add_ceiling_light(overlay, ceiling_center_z)
    _add_named_view_camera(overlay)
    _seed_viewport_pose(overlay)

    overlay.Save()
    print(f"Wrote: {OVERLAY_USD}")


if __name__ == "__main__":
    main()
    _app.close()
