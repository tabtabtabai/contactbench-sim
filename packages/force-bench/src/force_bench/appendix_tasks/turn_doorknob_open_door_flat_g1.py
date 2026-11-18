"""Task 28 "Turn doorknob to open door" with the Unitree G1 + Isaac Dex1 in place of the FR3 (embodiment variant,
branch g1-dex1; Clockwork "Embodiment variants", stage 2 first cut).

The scene is the row's own: the final kit door lying flat, latched, the deadbolt parked, the hidden lock probe.  The
table is moved so that its edge is in front of the G1 (the G1 stands on the floor, 0.74 m below the table top, and
its pelvis is welded at the whole-body controller's standing lean).  The arm follows an OPEN-LOOP joint trajectory
from matrix's MuJoCo G1 IK (``g1_door_knob_traj.json``): pre-grasp, descend to the knob's equator, squeeze, turn the
knob clockwise to its stop, lift the leaf about its hinge to 45 deg, hold.  It is a scripted reach-and-turn check on
real contact, not a closed-loop expert: nothing is servoed on measured state.

The verdict applies the row's criteria on world state, as the policy judge does (locked_at_start,
unlocked_by_knob_then_opened, finger_grip_sustained from the latch clearing, motion_stopped_door_held_open), and the
record carries ``robot_variant: true``.

Run: python -m force_bench.appendix_tasks.turn_doorknob_open_door_flat_g1 --model g1_dex1_model.npz
     --traj g1_door_knob_traj.json [--name <output>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import force_bench.appendix_tasks._door_kit as dk
from force_bench.appendix_tasks._door_kit import (LOCK_PROBE_NM, LOCK_PROBE_S, HingeProbe, build_door, door_angle,
                                                  drawn_shapes, joint_reader, lock_probe_check)
from force_bench.appendix_tasks.common import appendix_rig
from force_bench.g1.robot import add_g1, q_indices
from force_bench.kit.mechanisms.doorknob_module import JAMB_CLEAR_MM
from force_bench.rig import make_solver

NAME = "turn_doorknob_open_door_flat_g1"
FINGER_OPEN, FINGER_CLOSE = -0.02, 0.0          # Dex1 finger joints: -0.02 = 83 mm open; 0.0 = 43 mm (< the knob)
IMPRATIO = 10.0
CAM_POS, CAM_TARGET = np.array([0.55, -0.55, 0.55]), np.array([-0.08, 0.20, 0.05])


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="g1_dex1_model.npz (exported from matrix sims/mujoco)")
    ap.add_argument("--traj", required=True, help="g1_door_knob_traj.json (the IK trajectory and placement)")
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    traj = json.loads(Path(a.traj).read_text())
    rig = appendix_rig(a.name or NAME)
    pelvis = wp.transform(wp.vec3(*traj["pelvis_cb"]), wp.quat_identity())
    g1 = add_g1(rig.builder, a.model, pelvis)
    # the table's near edge in front of the G1 (the door stays at its pose)
    dk.TABLE_C = (float(traj["pelvis_cb"][0]) + 0.12 + dk.TABLE_HALF, dk.TABLE_C[1])
    choice = {"Doorknob geometry": "Doorknob 1, 55mm diameter", "Amount of rotation to unlock": "60°",
              "Rotation force": "5mm spring"}
    h = build_door(rig, knob=choice, lock="90°", lock_params={"hold_retracted": True})
    hk = h._modules["doorknob-module"]
    knob, leaf = h.bodies["knob"], h.bodies["leaf"]
    # contact: only the Dex1 finger boxes touch the knob, through its grasp hull; nothing of the G1 touches the leaf
    g1_shapes = [s for ss in g1["shapes"].values() for s in ss]
    drawn = set(drawn_shapes(rig, knob, hk.info["grip_hull_shapes"]))
    for s in g1_shapes:
        for t in rig.builder.body_shapes.get(leaf, []) + list(drawn):
            rig.builder.add_shape_collision_filter_pair(s, t)
    finger_bodies = [f"g1/right_hand_Link{i}_{k}" for i in (1, 2) for k in (2, 3)]
    rig.hard_joint_limits = True
    rig.limit_dampratio = 0.5
    rig.finalize(probes=[f"{hk.label}/latch", f"{hk.label}/knob", "door/leaf", *finger_bodies],
                 cam_pos=CAM_POS, cam_target=CAM_TARGET)
    rig.solver = make_solver(rig.model, iterations=24, **{**rig.solver_options, "impratio": IMPRATIO})
    probe = HingeProbe(rig, h)
    jq = joint_reader(rig, h)
    qi = q_indices(rig.model, g1["joints"])
    arm_names = traj["arm_joints"]
    side = traj["arm"]
    fingers = [f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"]
    starts = rig.model.joint_target_q_start.numpy() if hasattr(rig.model, "joint_target_q_start") else None
    tq_index = {n: (int(starts[g1["joints"][n]]) if starts is not None else qi[n]) for n in [*arm_names, *fingers]}
    way = traj["waypoints"]
    ts = np.array([w[0] for w in way])
    Q = np.array([w[1] for w in way])
    modes = [w[2] for w in way]
    # start at the first waypoint, fingers open
    q0 = rig.s0.joint_q.numpy()
    for n, v in zip(arm_names, Q[0]):
        q0[qi[n]] = v
    for n in fingers:
        q0[qi[n]] = FINGER_OPEN
    rig.s0.joint_q.assign(q0)
    import newton
    newton.eval_fk(rig.model, rig.s0.joint_q, rig.s0.joint_qd, rig.s0)
    log = []

    def control(t):
        probe.set(-LOCK_PROBE_NM if LOCK_PROBE_S[0] <= t < LOCK_PROBE_S[1] else 0.0)
        qt = rig.control.joint_target_q.numpy()
        k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        s = float(np.clip((t - ts[k]) / max(1e-9, ts[k + 1] - ts[k]), 0, 1))
        q = Q[k] + s * (Q[k + 1] - Q[k])
        for n, v in zip(arm_names, q):
            qt[tq_index[n]] = v
        f = FINGER_CLOSE if modes[k + 1 if s > 0.5 else k] == "close" else FINGER_OPEN
        for n in fingers:
            qt[tq_index[n]] = f
        rig.control.joint_target_q.assign(qt)

    def extra(state, t):
        bq = state.body_q.numpy()
        q = state.joint_q.numpy()
        th, ph, x = door_angle(bq[leaf]), float(q[jq["knob"]]), float(q[jq["latch"]])
        pads = []
        for i in (1, 2):
            pads.append(sum(rig.probes[f"g1/{side}_hand_Link{i}_{k}"].history[-1]["force"] for k in (2, 3)))
        log.append((t, th, ph, x, pads[0], pads[1]))
        return {"door_deg": round(float(np.degrees(-th)), 3), "knob_deg": round(float(np.degrees(-ph)), 2),
                "latch_retract_mm": round(x * 1000, 3), "pad1_N": round(pads[0], 2), "pad2_N": round(pads[1], 2)}

    rec = rig.run(float(ts[-1]), control, extra)
    paths = rec.save("rollout", force_keys=["door_deg", "knob_deg", "latch_retract_mm", "pad1_N", "pad2_N"])
    print("saved:", paths)
    L = np.array(log)
    t_a, th_a, ph_a, x_a, p1, p2 = L.T
    open_deg = np.degrees(-th_a)
    crit = {"locked_at_start": lock_probe_check(t_a, th_a)}
    eng = x_a * 1000.0 > JAMB_CLEAR_MM
    t_unl = float(t_a[np.argmax(eng)]) if eng.any() else float("inf")
    pre = open_deg[t_a < t_unl]
    crit["unlocked_by_knob_then_opened"] = {
        "pass": bool(eng.any() and (pre.max() if len(pre) else 0.0) < 1.0 and open_deg[-1] >= 5.0),
        "t_latch_clear_s": None if not eng.any() else round(t_unl, 3),
        "peak_latch_retract_mm": round(float(x_a.max() * 1e3), 3), "jamb_clear_mm": JAMB_CLEAR_MM,
        "peak_knob_deg": round(float(np.degrees(-ph_a).max()), 1),
        "door_deg_before_latch_clear_max": round(float(pre.max()) if len(pre) else 0.0, 3),
        "final_door_deg": round(float(open_deg[-1]), 2), "peak_door_deg": round(float(open_deg.max()), 2)}
    w = t_a >= (t_unl if eng.any() else t_a[-1])
    roll = lambda f: np.array([f[max(0, i - 2):i + 1].max() for i in range(len(f))])  # noqa: E731
    gp = min(roll(p1[w]).min(), roll(p2[w]).min()) if w.any() else 0.0
    crit["finger_grip_sustained"] = {"pass": bool(eng.any() and gp > 2.0), "min_pad_over_3_frames_N": round(float(gp), 2),
                                     "mean_pads_N": [round(float(p1[w].mean()), 2), round(float(p2[w].mean()), 2)]
                                     if w.any() else None}
    dd = float(np.ptp(open_deg[t_a >= t_a[-1] - 1.0]))
    crit["motion_stopped_door_held_open"] = {"pass": bool(dd < 1.0 and open_deg[-1] >= 5.0),
                                             "door_range_last_1s_deg": round(dd, 3)}
    ok = all(v["pass"] for v in crit.values())
    record = {"test_id": "5.1", "benchmark": "ContactBench 1.0", "task": "turn_doorknob_open_door_flat",
              "robot": "Unitree G1 + Dex1 (Isaac)", "robot_variant": True, "sim": "Newton + MuJoCo Warp",
              "seed": None, "policy_id": "g1-scripted-ik-open-loop", "privileged": True, "mode": "lockstep",
              "success": bool(ok),
              "success_detail": ("all criteria passed" if ok else
                                 "failed: " + ", ".join(k for k, v in crit.items() if not v["pass"])),
              "t_wall_s": round(float(t_a[-1]), 3), "t_world_s": round(float(t_a[-1]), 3),
              "attempts": 1, "retry_events": [], "irreversible": [], "render_version": 1,
              "judge": {"criteria": crit}, "placement": {k: traj[k] for k in ("pelvis_cb", "arm", "waist_pitch",
                                                                             "gamma_deg", "orient")},
              "video": [str(Path(paths["video"]).name)] if "video" in paths else []}
    (rig.out_dir / "episode.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("success", "success_detail")}))
    print(json.dumps(crit["unlocked_by_knob_then_opened"]))
    return record


if __name__ == "__main__":
    wp.init()
    main()
