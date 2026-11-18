"""Task 28 "Turn doorknob to open door" with the Unitree G1 + Isaac Dex1 in place of the FR3 (embodiment variant,
branch g1-dex1; Clockwork "Embodiment variants", stage 2: a privileged scripted expert).

The scene is the row's own: the final kit door lying flat, latched, the deadbolt parked, the hidden lock probe, and
``--seed`` moving the door as the FR3 row does (the robot stays put).  The table is moved so that its edge is in
front of the G1, which stands on the floor 0.74 m below the table top with its pelvis welded at the whole-body
controller's standing lean.  The arm follows the joint trajectory of matrix's MuJoCo G1 IK for this placement and
seed (``g1_door_knob_traj*.json``, made by g1_dex1_prepare.py): it starts at the pre-grasp 8 cm above the knob,
descends to the knob's equator (the Dex1's fingertips reach 27 mm past the pad centre, so a lower grasp hits the knob
module's housing) and squeezes, on the clock; then, closed on MEASURED state like the FR3 expert, it turns the knob
clockwise with a bounded lead over the measured knob angle until the measured latch has cleared the jamb, and lifts
the leaf about its hinge with a bounded lead over the measured door angle to 45 deg, and holds.  The hand poses come
from IK tables over the knob angle and the door angle (MuJoCo, same model).  An open-loop turn outran the knob: the
pinch creeps under the spring torque in MuJoCo, as the FR3 task notes.

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
import os
from pathlib import Path

import numpy as np
import warp as wp

import force_bench.appendix_tasks._door_kit as dk
from force_bench.appendix_tasks._door_kit import (LOCK_PROBE_NM, LOCK_PROBE_S, HingeProbe, build_door, door_angle,
                                                  door_pose, drawn_shapes, joint_reader, lock_probe_check,
                                                  pads_check)
from force_bench.appendix_tasks.common import StopRun, appendix_rig
from force_bench.g1.robot import (SumProbe, add_g1, apply_look, camera_mounts, floor_and_table, init_pose,
                                  q_indices)
from force_bench.kit.mechanisms.doorknob_module import JAMB_CLEAR_MM
from force_bench.rig import make_solver

NAME = "turn_doorknob_open_door_flat_g1"
FINGER_OPEN, FINGER_CLOSE = -0.02, 0.0          # Dex1 finger joints: -0.02 = 83 mm open; 0.0 = 43 mm (< the knob)
IMPRATIO = 10.0
PRE_HOLD = 0.6                                  # s at the pre-grasp before descending (covers the lock probe)
I_UP = 8                                        # the plan's first segment (straight up from the reset pose): 8 steps
# the closed-loop turn and lift (privileged, on MEASURED knob, latch and door state, as the FR3 expert):
TURN_RATE, TURN_LEAD = 30.0, 15.0               # deg/s of commanded knob turn; at most this far ahead of the knob
CLEAR_MARGIN_MM = 0.1                           # turn until the measured latch is this far past the jamb face
PAUSE_S = 0.4
LIFT_RATE, LIFT_LEAD = 12.0, 3.0                # deg/s of commanded door angle; at most this far ahead of the door
OPEN_TO_DEG, HOLD_S = 45.0, 1.6
TIMEOUT_S = float(os.environ.get("G1_MAX_T", 40.0))   # G1_MAX_T: a short pre-flight run
CAM_POS, CAM_TARGET = np.array([0.55, -0.55, 0.55]), np.array([-0.08, 0.20, 0.05])


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="g1_dex1_model.npz (exported from matrix sims/mujoco)")
    ap.add_argument("--traj", required=True, help="the IK trajectory and placement for this seed")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--part", choices=["knob", "handle"], default="knob",
                    help="the doorknob (task 28) or the lever handle (task 29); the same door, spring and expert")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    traj = json.loads(Path(a.traj).read_text())
    if traj.get("seed") != a.seed:
        raise SystemExit(f"{a.traj} is for seed {traj.get('seed')}, not {a.seed}")
    rig = appendix_rig(a.name or (NAME if a.part == "knob" else "turn_door_handle_open_door_flat_g1"))
    pelvis = wp.transform(wp.vec3(*traj["pelvis_cb"]), wp.quat_identity())
    g1 = add_g1(rig.builder, a.model, pelvis)
    rig.robot_base = pelvis
    # the table's near edge in front of the G1; the door at the row's (seeded) pose
    dk.TABLE_C = (float(traj["pelvis_cb"][0]) + 0.12 + dk.TABLE_HALF, dk.TABLE_C[1])
    choice = ({"Doorknob geometry": "Doorknob 1, 55mm diameter", "Amount of rotation to unlock": "60°",
               "Rotation force": "5mm spring"} if a.part == "knob" else
              {"Door handle geometry": "Handle 1", "Amount of rotation to unlock": "60°",
               "Rotation force": "5mm spring"})
    pose, shift = door_pose(a.seed)
    h = build_door(rig, knob=choice, lock="90°", lock_params={"hold_retracted": True}, pose=pose)
    if "floor_cb" in traj:                       # the G1 on its floor, a solid table under the board (Isaac setup)
        floor_and_table(rig.builder, float(traj["floor_cb"]), dk.TABLE_C, (dk.TABLE_HALF, dk.TABLE_HALF))
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
    apply_look(rig.model, g1)                    # the G1's own colours, after the recorder's palette (finalize)
    rig.solver = make_solver(rig.model, iterations=24, **{**rig.solver_options, "impratio": IMPRATIO})
    probe = HingeProbe(rig, h)
    jq = joint_reader(rig, h)
    qi = q_indices(rig.model, g1["joints"])
    tqs = rig.model.joint_target_q_start.numpy()
    arm_names = traj["arm_joints"]
    fingers = [f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"]
    tq_index = {n: int(tqs[g1["joints"][n]]) for n in [*arm_names, *fingers]}
    tips = [g1["bodies"][f"{side}_hand_Link{i}_3"] for i in (1, 2)]
    # the approach on the clock: from the reset pose (held while the lock probe runs) up, across and down to the
    # squeeze.  Older trajectories started at the arm's home pose, inside the door: those start at the pre-grasp.
    # Isaac setup: start with the active hand already raised to the hover height (the end of the plan's straight-up
    # segment, I_UP).  At the reset pose that hand sits inside the door frame's near edge when the knob is ~50 cm
    # out: starting there drove the pads to 1.4 kN in 2.6 s.  Recorded as start_raised.
    first = traj.get("i_up", I_UP) if "floor_cb" in traj else 1
    way = traj["waypoints"][first:traj["i_close"] + 1]
    t0 = way[0][0]
    way = [(0.0, way[0][1], way[0][2])] + [(w[0] - t0 + PRE_HOLD, w[1], w[2]) for w in way]
    ts = np.array([w[0] for w in way])
    Q = np.array([w[1] for w in way])
    G = np.array([traj["tau_g"][first]] + traj["tau_g"][first:traj["i_close"] + 1])
    ret = traj.get("retreat") or []
    RQ = np.array([w[1] for w in ret]) if ret else None
    RG = np.array(traj.get("retreat_tau_g") or [])
    RT = np.array([w[0] for w in ret]) if ret else None
    partial = not traj["lift_table"]["deg"]
    modes = [w[2] for w in way]
    T_turn = {k: np.asarray(v) for k, v in traj["turn_table"].items()}
    T_lift = {k: np.asarray(v) for k, v in traj["lift_table"].items()}
    qds = rig.model.joint_qd_start.numpy()
    f_index = [int(qds[g1["joints"][n]]) for n in arm_names]
    clear_m = (JAMB_CLEAR_MM + CLEAR_MARGIN_MM) * 1e-3
    q0 = rig.s0.joint_q.numpy()
    qt0 = rig.control.joint_target_q.numpy()
    for n, v in init_pose(a.model).items():       # both arms in the reset pose; the idle one holds it throughout
        if n in g1["joints"]:
            q0[qi[n]], qt0[int(tqs[g1["joints"][n]])] = v, v
    for n, v in zip(arm_names, Q[0]):
        q0[qi[n]], qt0[tq_index[n]] = v, v
    for n in fingers:
        q0[qi[n]], qt0[tq_index[n]] = FINGER_OPEN, FINGER_OPEN
    rig.s0.joint_q.assign(q0)
    rig.control.joint_target_q.assign(qt0)
    import newton
    newton.eval_fk(rig.model, rig.s0.joint_q, rig.s0.joint_qd, rig.s0)
    log, tcp_log = [], []
    st = {"phase": "approach", "psi": 0.0, "phi": 0.0, "t_phase": 0.0, "phases": []}
    dt = 1.0 / rig.fps

    def table(T, x):
        x = float(np.clip(x, T["deg"][0], T["deg"][-1]))
        return (np.array([np.interp(x, T["deg"], T["q"][:, j]) for j in range(T["q"].shape[1])]),
                np.array([np.interp(x, T["deg"], T["tau_g"][:, j]) for j in range(T["tau_g"].shape[1])]))

    def go(phase, t):
        st["phase"], st["t_phase"] = phase, t
        st["phases"].append((round(t, 3), phase))

    def control(t):
        probe.set(-LOCK_PROBE_NM if LOCK_PROBE_S[0] <= t < LOCK_PROBE_S[1] else 0.0)
        bq = rig.s0.body_q.numpy()
        q_now = rig.s0.joint_q.numpy()
        knob_deg = float(np.degrees(-q_now[jq["knob"]]))
        latch = float(q_now[jq["latch"]])
        door_deg = float(np.degrees(-door_angle(bq[leaf])))
        ph = st["phase"]
        close = True
        if ph == "approach":
            k = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2))
            s_ = float(np.clip((t - ts[k]) / max(1e-9, ts[k + 1] - ts[k]), 0, 1))
            q, g = Q[k] + s_ * (Q[k + 1] - Q[k]), G[k] + s_ * (G[k + 1] - G[k])
            close = modes[k + 1] == "close"
            if t >= ts[-1]:
                go("turn", t)
        elif ph in ("turn", "pause"):
            if latch < clear_m:                   # still in the strike: advance, bounded ahead of the knob
                st["psi"] = min(st["psi"] + TURN_RATE * dt, knob_deg + TURN_LEAD, T_turn["deg"][-1])
                if partial and st["psi"] >= T_turn["deg"][-1] - 1e-6 and ph == "turn":
                    go("pause", t)                # one grasp reaches no further (a partial plan): stop turning
            elif ph == "turn":
                go("pause", t)
            q, g = table(T_turn, st["psi"])
            if ph == "pause" and t - st["t_phase"] >= PAUSE_S + (1.0 if partial else 0.0):
                go("release" if partial else "lift", t)
        elif ph == "release":                     # partial: open where the turn ended
            q, g = table(T_turn, st["psi"])
            close = False
            if t - st["t_phase"] >= 0.5:
                st["psi_open"] = st["psi"]
                go("back", t)
        elif ph == "back":                        # turn the open hand back to the grasp angle
            u = min(1.0, (t - st["t_phase"]) / 1.0)
            q, g = table(T_turn, st["psi_open"] * (1.0 - u))
            close = False
            if u >= 1.0:
                go("retreat", t)
        elif ph in ("retreat", "still"):          # straight up, back over the reset pose's hand, down into it
            tau_r = min(t - st["t_phase"], RT[-1]) if ph == "retreat" else RT[-1]
            k = int(np.clip(np.searchsorted(RT, tau_r, side="right") - 1, 0, len(RT) - 2))
            s_ = float(np.clip((tau_r - RT[k]) / max(1e-9, RT[k + 1] - RT[k]), 0, 1))
            q, g = RQ[k] + s_ * (RQ[k + 1] - RQ[k]), RG[k] + s_ * (RG[k + 1] - RG[k])
            close = False
            if ph == "retreat" and tau_r >= RT[-1]:
                go("still", t)
            if ph == "still" and t - st["t_phase"] >= 1.0:
                raise StopRun
        elif ph in ("lift", "hold"):
            if latch < clear_m and door_deg < 5.0:   # the latch slipped back before the door opened: turn again
                st["psi"] = min(st["psi"] + TURN_RATE * dt, knob_deg + TURN_LEAD, T_turn["deg"][-1])
            open_to = min(OPEN_TO_DEG, float(T_lift["deg"][-1]))   # as far as the arm reaches (>= 10 deg)
            st["phi"] = min(st["phi"] + LIFT_RATE * dt, door_deg + LIFT_LEAD, open_to)
            q, g = table(T_lift, st["phi"])
            if ph == "lift" and st["phi"] >= open_to and door_deg >= open_to - 1.0:
                go("hold", t)
            if ph == "hold" and t - st["t_phase"] >= HOLD_S:
                raise StopRun
        if t >= TIMEOUT_S:
            raise StopRun
        qt = rig.control.joint_target_q.numpy()
        for n, v in zip(arm_names, q):
            qt[tq_index[n]] = v
        f = FINGER_CLOSE if close else FINGER_OPEN
        for n in fingers:
            qt[tq_index[n]] = f
        rig.control.joint_target_q.assign(qt)
        jf = rig.control.joint_f.numpy()
        jf[f_index] = g
        rig.control.joint_f.assign(jf)

    def extra(state, t):
        bq = state.body_q.numpy()
        q = state.joint_q.numpy()
        th, ph, x = door_angle(bq[leaf]), float(q[jq["knob"]]), float(q[jq["latch"]])
        log.append((t, th, ph, x))
        tcp_log.append((t, *(0.5 * (bq[tips[0], :3] + bq[tips[1], :3]))))
        p1, p2 = (rig.probes[lb].history[-1]["force"] for lb in rig.pad_probes)
        return {"door_deg": round(float(np.degrees(-th)), 3), "knob_deg": round(float(np.degrees(-ph)), 2),
                "latch_retract_mm": round(x * 1000, 3), "pad1_N": round(p1, 2), "pad2_N": round(p2, 2)}

    try:
        rec = rig.run(TIMEOUT_S + 1.0, control, extra)
    except StopRun:
        rec = rig.rec
    print("phases:", st["phases"])
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
