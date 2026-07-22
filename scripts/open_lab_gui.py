"""Exec-script for isaac-sim.sh: open the volcaniarm lab stage and press Play."""
import omni.usd
import omni.timeline

USD = "/home/tamir/projects/volcaniarm_isaaclab/assets/usd/volcaniarm_lab.usd"

ok = omni.usd.get_context().open_stage(USD)
print(f"[open_lab] open_stage({USD}) -> {ok}", flush=True)
if ok:
    omni.timeline.get_timeline_interface().play()
    print("[open_lab] timeline playing — ROS2 graph live", flush=True)
