# volcaniarm_isaaclab

Isaac Lab training environment for the **volcaniarm** — a 2-DOF stepper-
driven 5-bar planar mechanism used for weed detection in agriculture.

Policy learns inverse kinematics via RL (no analytical IK).

Paired with the ROS2 runtime repo
[`volcaniarm_ws`](https://github.com/LevinTamir/volcaniarm_ws).

## Layout

```
assets/
├── urdf/                  # URDF + STL meshes (source of truth for USD)
└── usd/                   # GITIGNORED — regenerated from URDF
scripts/
├── convert_urdf.py        # URDF → base USD (instanceable meshes)
└── close_loop.py          # base USD + closure_joint overlay → closed USD
source/                    # IsaacLab external-project package (TBD)
```

## Regenerating the USDs

```bash
conda activate isaaclab_env
~/isaac/IsaacLab/isaaclab.sh -p scripts/convert_urdf.py
~/isaac/IsaacLab/isaaclab.sh -p scripts/close_loop.py
```

Produces `assets/usd/volcaniarm.usd` (base) and `assets/usd/volcaniarm_closed.usd`
(overlay with loop closure). IsaacLab training uses the closed variant.

## Joint-name contract with `volcaniarm_ws`

The closure joint is named `closure_joint` to match the passive-joint list
in `volcaniarm_controller/config/volcaniarm_controllers.yaml`. Don't rename
without updating the controller side.
