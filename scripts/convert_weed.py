"""Convert the 3D-printed fake-weed STL into a USD prop for Isaac Lab.

Produces:
    assets/usd/fake_weed.usd     (11.5 cm tall, origin at the canopy apex)

Source is `assets/STLs/fake_weed.stl` — the small print, authored in
millimetres at 73.35 mm tall. Only one STL is tracked: the "medium" variant
upstream is the identical mesh at exactly 2x, so any size is reachable by
scaling this one.

The source is ~386k triangles. The vision policy only ever sees the weed's
green *silhouette* (see `reach_vision_ame/mdp/green_mask.py`), so fine leaf
detail carries no signal and we decimate hard.

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/convert_weed.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import trimesh

from pathlib import Path

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

PROJECT = Path(__file__).resolve().parent.parent
USD_DIR = PROJECT / "assets/usd"
STL_DIR = PROJECT / "assets/STLs"
BUILD_DIR = USD_DIR / "_build"

# Source STL and the real-world height to scale it to.
#
# 0.115 m is ~1.57x the authored 73.35 mm — a print you can actually make,
# while staying reachable. `scripts/check_workspace.py` shows the EE bottoms
# out at z=0.052 m: the arm physically cannot reach the mat at any joint angle,
# so the reach target is the weed's *canopy*, and the reachable Y span depends
# strongly on how high that canopy sits (elbows within the reward's +-75 deg):
#     canopy 0.075 m -> Y span 0.20 m      canopy 0.20 m -> Y span 0.58 m
#     canopy 0.115 m -> Y span 0.25 m      canopy >0.30 m -> unreachable
# At 0.115 m the canopy clears the EE floor by ~6 cm and Y spans ~+-0.10.
SOURCES = {
    "fake_weed": (STL_DIR / "fake_weed.stl", 0.115),
}

# The USD is geometry only — the green material is bound at spawn time via
# `UsdFileCfg(visual_material=...)` in `reach_vision_ame_env_cfg.py`, which
# keeps the colour in config (and makes per-env colour randomisation a
# one-line change later). `MeshConverterCfg` has no visual_material field.

# Chosen by sweeping decimation targets and measuring silhouette IoU against
# the undecimated mesh (orthographic side/front rasters at 96x96 — the same
# resolution the policy sees). The knee:
#     4k faces -> IoU 0.86/0.84      30k -> 0.94/0.93
#     15k      -> IoU 0.92/0.90      60k -> 0.96/0.96
# 15k keeps ~0.91 while dropping 96% of the geometry. Below ~8k, clustering
# erodes the thin leaves and shrinks the silhouette ~12%, which would bias
# the apparent blob size the policy trains on.
TARGET_FACES = 15000


def decimate_vertex_clustering(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce face count by merging vertices that share a voxel cell.

    We roll this by hand rather than use `simplify_quadric_decimation`, which
    needs the optional `fast_simplification` backend that isn't installed in
    `isaaclab_env`. Clustering is cruder than quadric collapse, but it
    preserves the outline — the only property the green mask actually reads —
    and it adds no dependency.

    The grid pitch is solved by bisection because face count falls off
    non-linearly with cell size.
    """
    lo, hi = mesh.bounds
    diag = float(np.linalg.norm(hi - lo))

    def cluster(pitch: float) -> trimesh.Trimesh:
        # Quantise to a grid, then average the vertices landing in each cell.
        keys = np.floor((mesh.vertices - lo) / pitch).astype(np.int64)
        inverse = np.unique(keys, axis=0, return_inverse=True)[1].ravel()
        n_cells = int(inverse.max()) + 1
        verts = np.zeros((n_cells, 3))
        counts = np.bincount(inverse, minlength=n_cells)[:, None]
        np.add.at(verts, inverse, mesh.vertices)
        verts /= np.maximum(counts, 1)

        faces = inverse[mesh.faces]
        # Drop faces that collapsed to a line or a point.
        keep = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        out = trimesh.Trimesh(vertices=verts, faces=faces[keep], process=True)
        out.remove_unreferenced_vertices()
        return out

    # Bisect on pitch: bigger pitch -> fewer faces.
    lo_p, hi_p = diag * 1e-4, diag * 0.25
    best = cluster(lo_p)
    for _ in range(40):
        mid = 0.5 * (lo_p + hi_p)
        cand = cluster(mid)
        if len(cand.faces) > target_faces:
            lo_p = mid
        else:
            hi_p = mid
            best = cand
        if abs(len(cand.faces) - target_faces) < target_faces * 0.05:
            best = cand
            break
    return best


def prepare_mesh(src: Path, target_height_m: float) -> tuple[trimesh.Trimesh, dict]:
    """Load, decimate, scale to `target_height_m`, and put the origin at the apex.

    Two placement details matter:

    1. The STL sits in the positive octant with `bbox min == [0,0,0]` — the
       origin is a corner, not the centroid — so we recentre in X/Y.

    2. The origin goes at the mesh **apex**, not its base. The arm cannot
       reach the mat (EE floor is z=0.052 m), so the reach target is the
       weed's canopy. Putting the origin there means the prim's root pose *is*
       the target: spawning at z=height seats the base on the mat at z=0 while
       `position_weed_error` / `weed_pos_in_base` keep working against
       root_pos_w unchanged, with no reward-side offset to keep in sync.
       The trade-off is that this couples the asset to its spawn height — any
       spawn-time rescale must adjust spawn z by the same factor or the base
       leaves the mat.
    """
    mesh = trimesh.load(src, force="mesh")
    before = len(mesh.faces)

    mesh = decimate_vertex_clustering(mesh, TARGET_FACES)

    lo, hi = mesh.bounds
    mesh.apply_scale(target_height_m / float(hi[2] - lo[2]))

    lo, hi = mesh.bounds
    centre_xy = 0.5 * (lo + hi)
    mesh.apply_translation((-centre_xy[0], -centre_xy[1], -hi[2]))

    lo, hi = mesh.bounds
    stats = {
        "faces_before": before,
        "faces_after": len(mesh.faces),
        "height_m": float(hi[2] - lo[2]),
        "footprint_m": (float(hi[0] - lo[0]), float(hi[1] - lo[1])),
        "apex_z": float(hi[2]),
        "base_z": float(lo[2]),
    }
    return mesh, stats


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    for name, (src, target_h) in SOURCES.items():
        if not src.exists():
            raise FileNotFoundError(f"missing source STL: {src}")

        mesh, stats = prepare_mesh(src, target_h)
        obj_path = BUILD_DIR / f"{name}.obj"
        mesh.export(obj_path)

        cfg = MeshConverterCfg(
            asset_path=str(obj_path),
            usd_dir=str(USD_DIR),
            usd_file_name=f"{name}.usd",
            force_usd_conversion=True,
            # MUST stay False. `make_instanceable=True` pushes geometry into a
            # *shared* `<usd_dir>/Props/instanceable_meshes.usd`, so repeated
            # conversions silently overwrite each other's meshes. At 15k faces
            # the geometry is ~130 KB inline; env cloning still instances it.
            make_instanceable=False,
            # The physics schemas MUST be baked in here, not left to the env
            # cfg. `UsdFileCfg` calls `schemas.modify_*`, which only touches
            # prims that already carry the schema — it cannot *apply*
            # RigidBodyAPI. Without this the scene fails at init with
            # "Failed to find a rigid body when resolving .../Weed".
            #
            # Kinematic: the weed is a reach target that we teleport on reset
            # via `reset_root_state_uniform`. It must never respond to forces
            # or be knocked around by the arm mid-episode.
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=schemas_cfg.MassPropertiesCfg(mass=0.05),
            # A collider has to exist for PhysX to register the body, but it
            # stays disabled — the EE is meant to reach *to* the weed, not
            # collide with it. Bounding-cube approximation avoids a convex
            # decomposition over 14k faces that would never be used.
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=False),
            mesh_collision_props=schemas_cfg.BoundingCubePropertiesCfg(),
        )
        MeshConverter(cfg)

        print(
            f"{name}: {stats['faces_before']} -> {stats['faces_after']} faces, "
            f"height {stats['height_m'] * 100:.1f} cm, "
            f"footprint {stats['footprint_m'][0] * 100:.1f}x{stats['footprint_m'][1] * 100:.1f} cm, "
            f"apex_z {stats['apex_z']:.4f} (origin), base_z {stats['base_z']:.4f}"
        )
        print(f"  -> {USD_DIR / (name + '.usd')}")


if __name__ == "__main__":
    main()
    simulation_app.close()
