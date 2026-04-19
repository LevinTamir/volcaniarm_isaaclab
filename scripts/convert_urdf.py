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

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

PROJECT = Path(__file__).resolve().parent.parent
URDF = PROJECT / "assets/urdf/urdf/volcaniarm.urdf"
USD_DIR = PROJECT / "assets/usd"
USD_NAME = "volcaniarm.usd"


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


if __name__ == "__main__":
    main()
    simulation_app.close()
