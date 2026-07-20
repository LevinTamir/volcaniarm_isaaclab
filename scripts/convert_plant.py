"""Convert the 3D-printed fake-plant STLs into USD props for Isaac Lab.

Produces:
    assets/usd/fake_plant_small.usd    (~7.3 cm tall)
    assets/usd/fake_plant_medium.usd   (~14.3 cm tall)

The source STLs are ~386k triangles / 19 MB each — unusable as-is when
instanced across 512 training envs, and far too heavy to commit. The vision
policy only ever sees the plant's green *silhouette* (see the HSV mask in
`reach_vision_ame/mdp/green_mask.py`), so fine leaf detail carries no signal.
We decimate hard and keep the outline.

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/convert_plant.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import trimesh

from pathlib import Path

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

PROJECT = Path(__file__).resolve().parent.parent
USD_DIR = PROJECT / "assets/usd"
BUILD_DIR = USD_DIR / "_build"

# Source STLs. Authored in millimetres; `meduim` typo is upstream.
SOURCES = {
    "fake_plant_small": Path.home() / "Downloads/small_size_fake_plant_with_magnets.STL",
    "fake_plant_medium": Path.home() / "Downloads/meduim_size_fake_plant_with_magnets.STL",
}

MM_TO_M = 0.001

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

# Matches the real 3D print. Re-measure from a RealSense frame and update
# alongside GREEN_NOMINAL in reach_vision_ame/contract.py — the mask
# thresholds and this colour have to describe the same object.
PLANT_COLOR = (0.10, 0.62, 0.38)


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
        keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
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


def prepare_mesh(src: Path) -> tuple[trimesh.Trimesh, dict]:
    """Load, decimate, scale to metres, and seat the mesh on z=0.

    The STLs sit in the positive octant with `bbox min == [0,0,0]` — the
    origin is a corner, not the centroid. Left alone the plant would hover
    off to one side of its spawn point, so we recentre in X/Y and drop the
    base to z=0 so it rests *on* the mat.
    """
    mesh = trimesh.load(src, force="mesh")
    before = len(mesh.faces)

    mesh = decimate_vertex_clustering(mesh, TARGET_FACES)
    mesh.apply_scale(MM_TO_M)

    lo, hi = mesh.bounds
    centre_xy = 0.5 * (lo + hi)
    mesh.apply_translation((-centre_xy[0], -centre_xy[1], -lo[2]))

    lo, hi = mesh.bounds
    stats = {
        "faces_before": before,
        "faces_after": len(mesh.faces),
        "height_m": float(hi[2] - lo[2]),
        "footprint_m": (float(hi[0] - lo[0]), float(hi[1] - lo[1])),
        "base_z": float(lo[2]),
    }
    return mesh, stats


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    for name, src in SOURCES.items():
        if not src.exists():
            raise FileNotFoundError(f"missing source STL: {src}")

        mesh, stats = prepare_mesh(src)
        obj_path = BUILD_DIR / f"{name}.obj"
        mesh.export(obj_path)

        cfg = MeshConverterCfg(
            asset_path=str(obj_path),
            usd_dir=str(USD_DIR),
            usd_file_name=f"{name}.usd",
            force_usd_conversion=True,
            # MUST stay False. `make_instanceable=True` pushes geometry into a
            # *shared* `<usd_dir>/Props/instanceable_meshes.usd`, which is the
            # same file `convert_urdf.py` writes for the robot — so an
            # instanceable plant conversion silently overwrites the robot's
            # meshes (and the two plants clobber each other). At 15k faces the
            # geometry is ~130 KB inline; env cloning still instances it.
            make_instanceable=False,
            # Visual-only prop: the plant is a reach *target*, never touched.
            # No mass/rigid/collision props keeps it cheap and stops it being
            # knocked around mid-episode.
            collision_props=None,
            rigid_props=None,
            mass_props=None,
        )
        MeshConverter(cfg)

        print(
            f"{name}: {stats['faces_before']} -> {stats['faces_after']} faces, "
            f"height {stats['height_m'] * 100:.1f} cm, "
            f"footprint {stats['footprint_m'][0] * 100:.1f}x{stats['footprint_m'][1] * 100:.1f} cm, "
            f"base_z {stats['base_z']:.4f}"
        )
        print(f"  -> {USD_DIR / (name + '.usd')}")


if __name__ == "__main__":
    main()
    simulation_app.close()
