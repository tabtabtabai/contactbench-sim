"""Task 29 "Turn door handle to open door" with the Unitree G1 + Isaac Dex1 (embodiment variant, branch g1-dex1): the
doorknob variant's scene and closed-loop expert (``turn_doorknob_open_door_flat_g1``) with the row's lever handle 1.
The pads pinch across the bar 85 mm out from the spindle, so the lever is swept by a pad normal; the IK tables come
from g1_dex1_prepare_handle.py.

Run: python -m force_bench.appendix_tasks.turn_door_handle_open_door_flat_g1 --model g1_dex1_model.npz
     --traj g1_door_handle_traj.json [--seed N]
"""

from __future__ import annotations

import sys

import warp as wp

from force_bench.appendix_tasks.turn_doorknob_open_door_flat_g1 import main as _main


def main(argv=None):
    return _main([*(sys.argv[1:] if argv is None else argv), "--part", "handle"])


if __name__ == "__main__":
    wp.init()
    main()
