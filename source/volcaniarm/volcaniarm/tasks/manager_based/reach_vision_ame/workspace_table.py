# Copyright (c) 2026, Tamir Levin.
# SPDX-License-Identifier: Apache-2.0
"""AUTO-GENERATED -- DO NOT HAND-EDIT.

Reachable envelope measured by scripts/check_workspace.py --emit-table,
then clipped to the camera-visible interval by
scripts/check_weed_visibility.py --clip-table. Regenerate with those two
commands in that order (see each script's --help).
"""

ELBOW_LIMIT_RAD = 1.3089969390
GRID = 151
Z_STEP = 0.01
ARM_JOINT_RANGE_RAD = (-1.5651, 0.8726)
VISIBILITY_CLIPPED = True
# rows: (z_lo_world, z_hi_world, y_min, y_max)
TABLE = [
    (0.110, 0.120, -0.3449, 0.4625),
    (0.120, 0.130, -0.3543, 0.4682),
    (0.130, 0.140, -0.3720, 0.4792),
    (0.140, 0.150, -0.3852, 0.4898),
    (0.150, 0.160, -0.3983, 0.4950),
    (0.160, 0.170, -0.4111, 0.5050),
    (0.170, 0.180, -0.4200, 0.5099),
    (0.180, 0.190, -0.4325, 0.5193),
    (0.190, 0.200, -0.4448, 0.5239),
    (0.200, 0.210, -0.4569, 0.5328),
    (0.210, 0.220, -0.4602, 0.5371),
    (0.220, 0.230, -0.4720, 0.5453),
    (0.230, 0.240, -0.4835, 0.5493),
    (0.240, 0.250, -0.4917, 0.5570),
    (0.250, 0.260, -0.5029, 0.5607),
    (0.260, 0.270, -0.5139, 0.5677),
    (0.270, 0.280, -0.5166, 0.5711),
    (0.280, 0.290, -0.5273, 0.5744),
    (0.290, 0.300, -0.5351, 0.5807),
]
