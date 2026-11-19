"""Checked in for: the render-version comparison stills (clockwork-5 render version 2): one row's start state from its
own third-person camera and the policy LEFT and wrist cameras, at the process's render version.

    TB_RENDER_VERSION=2 python -m force_bench.policy_env.render_stills turn_doorknob_open_door_flat --out stills/v2

The scene is built exactly as ``force-bench run`` builds it (the row's main(), stopped at ``Rig.run`` through the
teleop UI's hook), so the start state, the camera poses, resolutions and fields of view are the row's own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp


def render(row: str, out: Path, args=()) -> dict:
    from force_bench import render_v2
    from force_bench.policy_env.cameras import PolicyCameras
    from force_bench.teleop.scenes import _capture_task
    import imageio.v3 as iio

    out.mkdir(parents=True, exist_ok=True)
    cap = _capture_task(row, tuple(args))
    rig = cap.rig
    rec = rig.rec
    rec.capture(rig.s0, 0.0)                         # the recorder's own frame: kit look, then the version's colors
    third = rec._img_host.numpy().copy()
    tf = rec.cam_tf.numpy()[0][0]
    cams = PolicyCameras(rig, scene_pos=np.array(tf[:3], float), scene_target=np.asarray(rec.cam2_target, float))
    views = cams.capture(0.0)
    paths = {"third_person": out / f"{row}_third.png", "left": out / f"{row}_left.png",
             "wrist": out / f"{row}_wrist.png"}
    iio.imwrite(paths["third_person"], third)
    iio.imwrite(paths["left"], views["images"]["left"])
    iio.imwrite(paths["wrist"], views["images"]["wrist"])
    meta = {"row": row, "render_version": render_v2.version(), "third_person_px": list(third.shape[:2]),
            "policy_px": list(views["images"]["left"].shape[:2]),
            "calibration": views["calibration"], "files": {k: v.name for k, v in paths.items()}}
    (out / f"{row}.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("row")
    ap.add_argument("--out", type=Path, required=True)
    a, rest = ap.parse_known_args(argv)
    wp.init()
    print(json.dumps({k: v for k, v in render(a.row, a.out, rest).items() if k != "calibration"}))


if __name__ == "__main__":
    main()
