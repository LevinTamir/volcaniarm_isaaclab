"""Convert the volcaniarm URDF into a USD asset for Isaac Lab.

Produces:
    assets/usd/volcaniarm.usd                 (articulation + physics, open tree)
    assets/usd/Props/instanceable_meshes.usd  (shared instanceable mesh prototypes)

Run with:
    conda activate isaaclab_env
    ~/isaac/IsaacLab/isaaclab.sh -p scripts/convert_urdf.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pathlib import Path

from pxr import Usd

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

PROJECT = Path(__file__).resolve().parent.parent
URDF = PROJECT / "assets/urdf/urdf/volcaniarm.urdf"
USD_DIR = PROJECT / "assets/usd"
USD_NAME = "volcaniarm.usd"


def _prune_dangling_visuals(usd_dir: Path) -> None:
    """Remove ``<link>/visuals`` prims that reference a non-existent target.

    The URDF importer authors a ``visuals`` Xform (referencing
    ``/visuals/<link>`` in the physics layer) for *every* link, but the
    physics layer only holds ``/visuals/<link>`` for links that actually
    carry a visual mesh. Frame-only links (the ee tips, the camera optical
    frame) get a dangling reference, which Isaac logs as a noisy
    "Unresolved reference prim path" warning on every stage load. Strip
    those empty visuals prims so composition stays clean.
    """
    cfg_dir = usd_dir / "configuration"
    base_path = cfg_dir / "volcaniarm_base.usd"
    phys_path = cfg_dir / "volcaniarm_physics.usd"
    if not base_path.exists() or not phys_path.exists():
        return

    phys = Usd.Stage.Open(str(phys_path))
    vis_scope = phys.GetPrimAtPath("/visuals")
    valid = {c.GetName() for c in vis_scope.GetChildren()} if vis_scope else set()

    base = Usd.Stage.Open(str(base_path))
    root = base.GetDefaultPrim()  # the articulation root, e.g. /volcaniarm
    # Collect paths first: RemovePrim recomposes the stage and expires any
    # UsdPrim handles fetched before it, so we must not remove mid-traversal.
    # Only target link-level `/<root>/<link>/visuals` prims — never the
    # top-level `/visuals` scope that holds the actual meshes.
    targets = [
        (prim.GetParent().GetName(), prim.GetPath())
        for prim in base.Traverse()
        if prim.GetName() == "visuals"
        and prim.GetParent().GetParent() == root
        and prim.GetParent().GetName() not in valid
    ]
    for _link, path in targets:
        base.RemovePrim(path)
    if targets:
        base.GetRootLayer().Save()
    print(f"Pruned dangling visuals for: {[link for link, _ in targets] or 'none'}")


def main() -> None:
    cfg = UrdfConverterCfg(
        asset_path=str(URDF),
        usd_dir=str(USD_DIR),
        usd_file_name=USD_NAME,
        force_usd_conversion=True,
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=False,
        collider_type="convex_hull",
        replace_cylinders_with_capsules=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=400.0,
                damping=40.0,
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"Wrote: {converter.usd_path}")
    _prune_dangling_visuals(USD_DIR)


if __name__ == "__main__":
    main()
    simulation_app.close()
