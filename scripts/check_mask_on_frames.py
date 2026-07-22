"""Run the deployed mask pipeline over real camera frames. No simulator needed.

Stage 2 of the sim-to-sim (and later sim-to-real) diagnosis. Takes frames saved
by `volcaniarm_ws/scripts/grab_camera_frames.py`, applies the EXACT resize the
C++ controller applies, then the EXACT mask functions baked into the exported
ONNX graph — so whatever this reports is what the policy actually sees.

`mask_ops` imports torch and nothing else, which is what makes this runnable
without launching Isaac.

Usage:
    conda activate isaaclab_env
    python scripts/check_mask_on_frames.py --frames /tmp/volcaniarm_frames
"""

import argparse
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
MASK_OPS = (
    PROJECT
    / "source/volcaniarm/volcaniarm/tasks/manager_based/reach_vision_ame/mask_ops.py"
)


def load_mask_ops():
    """Load by file path — importing the package would drag in all of IsaacLab."""
    spec = importlib.util.spec_from_file_location("mask_ops", MASK_OPS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/tmp/volcaniarm_frames")
    ap.add_argument("--cam-hw", type=int, nargs=2, default=(96, 96),
                    help="controller's image_height image_width")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mo = load_mask_ops()
    cam_h, cam_w = args.cam_hw
    out_dir = args.out or os.path.join(args.frames, "mask")
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(Path(args.frames).glob("frame_*.png"))
    if not paths:
        raise SystemExit(f"no frame_*.png in {args.frames} — run grab_camera_frames.py first")

    print(f"thresholds: {mo.GREEN_NOMINAL}")
    for p in paths:
        bgr = cv2.imread(str(p))
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Exactly what rl_vision_policy_controller.cpp does: INTER_AREA to the
        # configured size. Note this does NOT preserve aspect ratio — a 4:3
        # source is squashed into a square, which is the single most likely
        # sim-to-sim geometry mismatch.
        resized = cv2.resize(rgb_full, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

        t = torch.from_numpy(resized).unsqueeze(0)
        mask = mo.rgb_to_green_mask(t)
        iso = mo.isolate_blob(mask)

        m0, i0 = mask[0], iso[0]
        cov = float((i0 > 0.5).float().mean())
        # Blob centroid in normalised [-1, 1] coords — this is what the policy
        # keys on, so a wrong centroid means a wrong reach.
        if cov > 0:
            ys, xs = torch.nonzero(i0 > 0.5, as_tuple=True)
            cy = float(ys.float().mean()) / i0.shape[0] * 2 - 1
            cx = float(xs.float().mean()) / i0.shape[1] * 2 - 1
            centroid = f"centroid=({cx:+.3f},{cy:+.3f})"
        else:
            centroid = "centroid=NONE"

        print(
            f"{p.name}: src={rgb_full.shape[1]}x{rgb_full.shape[0]} "
            f"peak={float(m0.max()):.4f} coverage={cov:.4f} {centroid}"
        )

        panel = np.concatenate(
            [
                resized,
                np.stack([(m0.numpy() * 255).astype(np.uint8)] * 3, -1).repeat(2, 0).repeat(2, 1),
                np.stack([(i0.numpy() * 255).astype(np.uint8)] * 3, -1).repeat(2, 0).repeat(2, 1),
            ],
            axis=1,
        )
        cv2.imwrite(
            os.path.join(out_dir, f"{p.stem}_mask.png"),
            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
        )

    # HSV of green-dominant pixels — compare against GREEN_NOMINAL to see if
    # the renderer/camera has shifted the weed's colour off the band.
    allsel = []
    for p in paths:
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx, mn = rgb.max(-1), rgb.min(-1)
        d = mx - mn + 1e-6
        h = np.where(mx == r, np.where(g >= b, (g - b) / d, (g - b) / d + 6),
                     np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
        s, v = (mx - mn) / (mx + 1e-6), mx
        sel = (g > r * 1.25) & (g > b * 1.15) & (s > 0.25) & (v > 0.15)
        if sel.any():
            allsel.append(np.stack([h[sel], s[sel], v[sel]], -1))
    if allsel:
        a = np.concatenate(allsel)
        for i, nm in enumerate(("hue", "sat", "val")):
            q = np.percentile(a[:, i], [5, 50, 95])
            print(f"{nm}: p05={q[0]:.4f} median={q[1]:.4f} p95={q[2]:.4f}")
        print(f"OpenCV H median = {np.median(a[:, 0]) * 180:.1f}  "
              f"(band centre {mo.GREEN_NOMINAL['hue_center'] * 180:.1f} "
              f"+-{mo.GREEN_NOMINAL['hue_halfwidth'] * 180:.1f})")
    else:
        print("WARNING: no green-dominant pixels in any frame — weed not in view?")

    print(f"\nwrote panels to {out_dir}")


if __name__ == "__main__":
    main()
