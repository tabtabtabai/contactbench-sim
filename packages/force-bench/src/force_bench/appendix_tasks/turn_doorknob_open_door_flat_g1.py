"""Task 28 "Turn doorknob to open door" with the Unitree G1 + Isaac Dex1 in place of the FR3 (embodiment variant,
branch g1-dex1; Clockwork "Embodiment variants", stage 2: a privileged scripted expert).

The scene is the row's own: the final kit door lying flat, latched, the deadbolt parked, the hidden lock probe, and
``--seed`` moving the door as the FR3 row does (the robot stays put).  The table is moved so that its edge is in
front of the G1, which stands on the floor 0.74 m below the table top with its pelvis welded at the whole-body
controller's standing lean.  The arm follows the joint trajectory of matrix's MuJoCo G1 IK for this placement and
seed (``g1_door_knob_traj*.json``, made by g1_dex1_prepare.py): it starts at the pre-grasp 8 cm above the knob,
descends to the knob's equator (the Dex1's fingertips reach 27 mm past the pad centre, so a lower grasp hits the knob
module's housing), squeezes, turns the knob clockwise to its stop, lifts the leaf about its hinge to 45 deg and holds.

The callbacks keep the FR3 task's names and logs (``probe``, ``log``, ``tcp_log``), and the rig carries its pad
probes, cameras and root (``pad_probes``, ``camera_mounts``, ``robot_base``), so ``force_bench.policy_env`` runs it
with the row's own adapter and judge (DoorAdapter): ``run_row("module:force_bench.appendix_tasks.
turn_doorknob_open_door_flat_g1", seed, ExpertDriver(), extra_argv=[--model ..., --traj ...])``.  Run alone, it also
writes its own verdict on the same criteria.

Run: python -m force_bench.appendix_tasks.turn_doorknob_open_door_flat_g1 --model g1_dex1_model.npz
     --traj g1_door_knob_traj.json [--seed N] [--name <output>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import force_bench.appendix_tasks._door_kit as dk
from force_bench.appendix_tasks._door_kit import (LOCK_PROBE_NM, LOCK_PROBE_S, HingeProbe, build_door, door_angle,
                                                  door_pose, drawn_shapes, joint_reader, lock_probe_check,
                                                  pads_check)
from force_bench.appendix_tasks.common import appendix_rig
from force_bench.g1.robot import SumProbe, add_g1, camera_mounts, q_indices
from force_bench.kit.mechanisms.doorknob_module import JAMB_CLEAR_MM
from force_bench.rig import make_solver

NAME = "turn_doorknob_open_door_flat_g1"
FINGER_OPEN, FINGER_CLOSE = -0.02, 0.0          # Dex1 finger joints: -0.02 = 83 mm open; 0.0 = 43 mm (< the knob)
IMPRATIO = 10.0
PRE_HOLD = 0.6                                  # s at the pre-grasp before descending (covers the lock probe)
CAM_POS, CAM_TARGET = np.array([0.55, -0.55, 0.55]), np.array([-0.08, 0.20, 0.05])


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="g1_dex1_model.npz (exported from matrix sims/mujoco)")
    ap.add_argument("--traj", required=True, help="the IK trajectory and placement for this seed")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    traj = json.loads(Path(a.traj).read_text())
    if traj.get("seed") != a.seed:
        raise SystemExit(f"{a.traj} is for seed {traj.get('seed')}, not {a.seed}")
    rig = appendix_rig(a.name or NAME)
    pelvis = wp.transform(wp.vec3(*traj["pelvis_cb"]), wp.quat_identity())
    g1 = add_g1(rig.builder, a.model, pelvis)
    rig.robot_base = pelvis
    # the table's near edge in front of the G1; the door at the row's (seeded) pose
    dk.TABLE_C = (float(traj["pelvis_cb"][0]) + 0.12 + dk.TABLE_HALF, dk.TABLE_C[1])
    choice = {"Doorknob geometry": "Doorknob 1, 55mm diameter", "Amount of rotation to unlock": "60°",
              "Rotation force": "5mm spring"}
    pose, shift = door_pose(a.seed)
    h = build_door(rig, knob=choice, lock="90°", lock_params={"hold_retracted": True}, pose=pose)
    hk = h._modules["doorknob-module"]
    knob, leaf = h.bodies["knob"], h.bodies["leaf"]
    # contact: only the Dex1 finger boxes touch the knob, through its grasp hull; nothing of the G1 touches the leaf
    g1_shapes = [s for ss in g1["shapes"].values() for s in ss]
    drawn = set(drawn_shapes(rig, knob, hk.info["grip_hull_shapes"]))
    for s in g1_shapes:
        for t in rig.builder.body_shapes.get(leaf, []) + list(drawn):
            rig.builder.add_shape_collision_filter_pair(s, t)
    side = traj["arm"]
    links = {i: [f"g1/{side}_hand_Link{i}_{k}" for k in (2, 3)] for i in (1, 2)}
    rig.hard_joint_limits = True
    rig.limit_dampratio = 0.5
    rig.finalize(probes=[f"{hk.label}/latch", f"{hk.label}/knob", "door/leaf", *links[1], *links[2]],
                 cam_pos=CAM_POS + shift, cam_target=CAM_TARGET + shift)
    for i in (1, 2):                             # each pad = its finger's two links
        rig.probes[f"g1/pad{i}"] = SumProbe([rig.probes.pop(lb) for lb in links[i]])
    rig.pad_probes = ("g1/pad1", "g1/pad2")
    rig.camera_mounts = camera_mounts(a.model, side)
    rig.solver = make_solver(rig.model, iterations=24, **{**rig.solver_options, "impratio": IMPRATIO})
    probe = HingeProbe(rig, h)
    jq = joint_reader(rig, h)
    qi = q_indices(rig.model, g1["joints"])
    tqs = rig.model.joint_target_q_start.numpy()
    arm_names = traj["arm_joints"]
    fingers = [f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"]
    tq_index = {n: int(tqs[g1["joints"][n]]) for n in [*arm_names, *fingers]}
    tips = [g1["bodies"][f"{side}_hand_Link{i}_3"] for i in (1, 2)]
    way = traj["waypoints"][1:]                  # start AT the pre-grasp, clear of the door (see the docstring)
    t0 = way[0][0]
    way = [(0.0, way[0][1], way[0][2])] + [(w[0] - t0 + PRE_HOLD, w[1], w[2]) for w in way]
    ts = np.array([w[0] for w in way])
    Q = np.array([w[1] for w in way])
    modes = [w[2] for w in way]
    # the whole-body controller's gravity compensation: the arm's gravity torques at each waypoint, fed forward
    G = np.array([traj["tau_g"][1]] + traj["tau_g"][1:]) if "tau_g" in traj else np.zeros_like(Q)
    qds = rig.model.joint_qd_start.numpy()
    f_index = [int(qds[g1["joints"][n]]) for n in arm_names]
    q0 = rig.s0.joint_q.numpy()
    qt0 = rig.control.joint_target_q.numpy()
    for n, v in zip(arm_names, Q[0]):
        q0[qi[n]], qt0[tq_index[n]] = v, v
    for n in fingers:
        q0[qi[n]], qt0[tq_index[n]] = FINGER_OPEN, FINGER_OPEN
    rig.s0.joint_q.assign(q0)
    rig.control.joint_target_q.assign(qt0)
    import newton
    newton.eval_fk(rig.model, rig.s0.joint_q, rig.s0.joint_qd, rig.s0)
    log, tcp_log = [], []

    def control(t):
        probe.set(-LOCK_PROBE_NM if LOCK_PROBE_S[0] <= t < LOCK_PROBE_S[1] else 0.0)
        qt = rig.control.joint_target_q.numpy()
        k = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2))
        s = float(np.clip((t - ts[k]) / max(1e-9, ts[k + 1] - ts[k]), 0, 1))
        q = Q[k] + s * (Q[k + 1] - Q[k])
        for n, v in zip(arm_names, q):
            qt[tq_index[n]] = v
        jf = rig.control.joint_f.numpy()
        jf[f_index] = G[k] + s * (G[k + 1] - G[k])
        rig.control.joint_f.assign(jf)
        f = FINGER_CLOSE if modes[k + 1] == "close" else FINGER_OPEN
        for n in fingers:
            qt[tq_index[n]] = f
        rig.control.joint_target_q.assign(qt)

    def extra(state, t):
        bq = state.body_q.numpy()
        q = state.joint_q.numpy()
        th, ph, x = door_angle(bq[leaf]), float(q[jq["knob"]]), float(q[jq["latch"]])
        log.append((t, th, ph, x))
        tcp_log.append((t, *(0.5 * (bq[tips[0], :3] + bq[tips[1], :3]))))
        p1, p2 = (rig.probes[lb].history[-1]["force"] for lb in rig.pad_probes)
        return {"door_deg": round(float(np.degrees(-th)), 3), "knob_deg": round(float(np.degrees(-ph)), 2),
                "latch_retract_mm": round(x * 1000, 3), "pad1_N": round(p1, 2), "pad2_N": round(p2, 2)}

    rec = rig.run(float(ts[-1]), control, extra)
    paths = rec.save("rollout", force_keys=["door_deg", "knob_deg", "latch_retract_mm", "pad1_N", "pad2_N"])
    print("saved:", paths)
    # the row's criteria on world state (the same windows as policy_env's DoorAdapter)
    L = np.array(log)
    t_a, th_a, ph_a, x_a = L.T
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
    if eng.any():
        shut = np.flatnonzero(open_deg < 1.0)
        crit["finger_grip_sustained"] = pads_check(rig, max(t_unl, float(t_a[shut[-1]]) if len(shut) else 0.0),
                                                   float(t_a[-1]))
    else:
        crit["finger_grip_sustained"] = {"pass": False, "note": "the latch never cleared"}
    tcp = np.array(tcp_log)
    m = tcp[:, 0] >= t_a[-1] - 1.0
    drift = float(np.linalg.norm(tcp[m, 1:][-1] - tcp[m, 1:][0]) * 1000)
    dd = float(np.ptp(open_deg[t_a >= t_a[-1] - 1.0]))
    crit["motion_stopped_door_held_open"] = {"pass": bool(drift < 3.0 and dd < 1.0 and open_deg[-1] >= 5.0),
                                             "tcp_drift_last_1s_mm": round(drift, 2),
                                             "door_range_last_1s_deg": round(dd, 3)}
    ok = all(v["pass"] for v in crit.values())
    verdict = {"pass": bool(ok), "criteria": crit, "robot": "Unitree G1 + Dex1", "robot_variant": True,
               "placement": {k: traj[k] for k in ("pelvis_cb", "arm", "waist_pitch", "gamma_deg", "orient")},
               "seed": a.seed, "door_shift_mm": [round(float(v) * 1e3, 1) for v in shift[:2]]}
    (rig.out_dir / "g1_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps({"pass": verdict["pass"], "failed": [k for k, v in crit.items() if not v["pass"]]}))
    print(json.dumps(crit["unlocked_by_knob_then_opened"]))
    return verdict


if __name__ == "__main__":
    wp.init()
    main()
