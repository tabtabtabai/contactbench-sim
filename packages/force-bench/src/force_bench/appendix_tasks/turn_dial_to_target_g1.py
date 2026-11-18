"""Task 30 "Turn dial to target position" (the standard dial, flat knob, +90 deg) with the Unitree G1 + Isaac Dex1 in
place of the FR3 (embodiment variant, branch g1-dex1; Clockwork "Embodiment variants", stage 2: a privileged scripted
expert).

The scene is the row's own dial (``turn_dial_to_target --dial standard``: a six-pad friction brake, ~0.9 N.m to turn,
holds any angle) at its seeded start; the table is moved so that its edge is in front of the G1, which stands on the
floor 0.74 m below the table top with its pelvis welded at the whole-body controller's standing lean.  The expert:
on the clock, from the pre-grasp 8 cm above, descend to the FR3 expert's grip height (fingertips 5 mm above the grip's
base) and close across the wing bar's flats; then, closed on the MEASURED dial angle like the FR3 expert, turn
counter-clockwise with a bounded lead until the dial is within TRIM_DEG of the target; open, lift 8 cm, hold still.
Hand poses and gravity feed-forward come from IK tables (g1_dex1_prepare_dial.py, matrix's MuJoCo G1, same model).

``policy_env``'s DialAdapter judges it unchanged: the logger object (``extra``'s owner) has the FR3 expert's fields
(``log`` rows with t, ang, press, fL, fR, robot, tips; ``target``; ``dial``).

Run: python -m force_bench.appendix_tasks.turn_dial_to_target_g1 --model g1_dex1_model.npz --traj g1_dial_traj.json
     [--seed N] [--name <output>]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

from force_bench.appendix_tasks.common import RobotContactProbe, StopRun, appendix_rig, sample_start
from force_bench.appendix_tasks.turn_dial_to_target import (DIAL_KEYS, DIAL_PLACE, IMPRATIO, ITERS, KNOB_LABELS,
                                                            SUBSTEPS, T_STILL)
from force_bench.g1.robot import (SumProbe, add_g1, apply_look, camera_mounts, floor_and_table, init_pose,
                                  q_indices)
from force_bench.kit import add_asset, place_on_table
from force_bench.kit.mechanisms import dial as D
from force_bench.rig import make_solver
from force_bench.tasks.nut_bolt_arm import PairProbe

NAME = "turn_dial_to_target_g1"
FINGER_OPEN, FINGER_CLOSE = -0.02, 0.0245      # Dex1 finger joints: 83 mm open; fully closed (the squeeze is the
#                                                effort limit, G1_FINGER_EFFORT)
PRE_HOLD = 0.4
TURN_RATE, TURN_LEAD, TRIM_DEG = 40.0, 12.0, 2.0   # deg/s; at most this far ahead of the measured dial; done within
T_SETTLE, T_OPEN, T_RETREAT = 0.3, 0.5, 1.2
TIMEOUT_S = float(os.environ.get("G1_MAX_T", 40.0))   # G1_MAX_T: a short pre-flight run
CAM_POS, CAM_TARGET = np.array([0.30, -0.35, 0.30]), np.array([-0.05, 0.05, 0.04])


class G1DialLog:
    """The world-state logger the dial judge reads (the FR3 Expert's fields)."""

    def __init__(self, rig, h, probes, tips, target_rad):
        self.rig, self.h, self.p, self.tips = rig, h, probes, tips
        self.target, self.dial = float(target_rad), "standard"
        self.log = []

    def extra(self, state, t):
        ang, press = D.knob_coords(self.rig, self.h, state)
        fl = self.p["finL"].history[-1]["force"] if self.p["finL"].history else 0.0
        fr = self.p["finR"].history[-1]["force"] if self.p["finR"].history else 0.0
        ro = self.p["robot_object"].history[-1]["force"] if self.p["robot_object"].history else 0.0
        bq = state.body_q.numpy()
        tips = 0.5 * (bq[self.tips[0], :3] + bq[self.tips[1], :3])
        self.log.append({"t": t, "ang": float(ang), "press": float(press), "fL": fl, "fR": fr, "robot": ro,
                         "tips": tips.tolist()})
        return {"knob_deg": round(float(np.degrees(ang)), 2), "pads_N": round(fl + fr, 2),
                "robot_object_contact_N": round(ro, 4)}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    traj = json.loads(Path(a.traj).read_text())
    if traj.get("seed") != a.seed:
        raise SystemExit(f"{a.traj} is for seed {traj.get('seed')}, not {a.seed}")
    start = sample_start({"dial": DIAL_PLACE}, a.seed)["dial"]
    if not np.allclose(start, traj["dial_start"], atol=1e-9):
        raise SystemExit(f"{a.traj}: dial start {traj['dial_start']} != the row's {start}")
    key, choice = DIAL_KEYS["standard"], {"Knob geometry": KNOB_LABELS["flat"]}
    rig = appendix_rig(a.name or NAME)
    rig.substeps = SUBSTEPS
    rig.collide_every_substep = True
    pelvis = wp.transform(wp.vec3(*traj["pelvis_cb"]), wp.quat_identity())
    g1 = add_g1(rig.builder, a.model, pelvis)
    rig.robot_base = pelvis
    px = float(traj["pelvis_cb"][0])
    rig.table(center_xy=(px + 0.12 + 0.35, float(start[1])), half=0.35)
    if "floor_cb" in traj:                       # the G1 on its floor, a solid table under the board (Isaac setup)
        floor_and_table(rig.builder, float(traj["floor_cb"]), (px + 0.12 + 0.35, float(start[1])), (0.35, 0.35))
    pose = place_on_table(key, choice, xy=(start[0], start[1]), yaw=float(start[2]))
    h = add_asset(rig, key, choice, pose=pose)
    side = traj["arm"]
    links = {i: [f"g1/{side}_hand_Link{i}_{k}" for k in (2, 3)] for i in (1, 2)}
    shift = np.array([start[0], start[1], 0.0])
    rig.finalize(probes=[f"{h.label}/knob"], cam_pos=CAM_POS + shift, cam_target=CAM_TARGET + shift)
    rig.solver = make_solver(rig.model, iterations=ITERS, impratio=IMPRATIO)
    labels = [str(lbl) for lbl in rig.model.shape_label]
    knob_shapes = [lbl for lbl in labels if lbl.startswith(f"kit/{h.label}/") and "knob" in lbl]
    static_labels = [lbl for lbl in labels if lbl.startswith(f"kit/{h.label}/")
                     and any(s in lbl for s in ("_top", "_bottom", "gate_"))]
    moving = [rig.model.body_label[b] for b in h.bodies.values()]
    probes = {"finL": SumProbe([PairProbe(rig.model, lb, counterpart_shapes=knob_shapes) for lb in links[1]]),
              "finR": SumProbe([PairProbe(rig.model, lb, counterpart_shapes=knob_shapes) for lb in links[2]]),
              "robot_object": RobotContactProbe(rig.model, moving, static_shapes=static_labels)}
    rig.probes.update(probes)
    rig.pad_probes = ("finL", "finR")
    rig.camera_mounts = camera_mounts(a.model, side)
    apply_look(rig.model, g1)                    # the G1's own colours, after the recorder's palette (finalize)
    tips = [g1["bodies"][f"{side}_hand_Link{i}_3"] for i in (1, 2)]
    lg = G1DialLog(rig, h, probes, tips, np.radians(traj["target_deg"]))
    qi = q_indices(rig.model, g1["joints"])
    tqs = rig.model.joint_target_q_start.numpy()
    qds = rig.model.joint_qd_start.numpy()
    arm_names = traj["arm_joints"]
    fingers = [f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"]
    tq_index = {n: int(tqs[g1["joints"][n]]) for n in [*arm_names, *fingers]}
    f_index = [int(qds[g1["joints"][n]]) for n in arm_names]
    way = traj["waypoints"][:traj.get("i_close", len(traj["waypoints"]) - 1) + 1]
    ts = np.array([w[0] for w in way]) + PRE_HOLD
    ts[0] = 0.0
    Q, G = np.array([w[1] for w in way]), np.array(traj["tau_g"][:len(way)])
    ret = traj.get("retreat") or []              # straight up, back over the reset pose's hand, down into it
    RQ = np.array([w[1] for w in ret]) if ret else None
    RG = np.array(traj.get("retreat_tau_g") or [])
    RT = np.array([w[0] for w in ret]) if ret else None
    modes = [w[2] for w in way]
    T_turn = {k: np.asarray(v) for k, v in traj["turn_table"].items()}
    T_ret = {k: np.asarray(v) for k, v in traj["retreat_table"].items()} if "retreat_table" in traj else None
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
    dt = 1.0 / rig.fps
    target_deg = float(traj["target_deg"])
    st = {"phase": "approach", "cmd": 0.0, "t_phase": 0.0, "phases": [], "q_end": None}

    def interp(T, key, x):
        xs = T[key]
        x = float(np.clip(x, xs[0], xs[-1]))
        return (np.array([np.interp(x, xs, T["q"][:, j]) for j in range(T["q"].shape[1])]),
                np.array([np.interp(x, xs, T["tau_g"][:, j]) for j in range(T["tau_g"].shape[1])]))

    def go(phase, t):
        st["phase"], st["t_phase"] = phase, t
        st["phases"].append((round(t, 3), phase))

    def control(t):
        ph = st["phase"]
        ang = float(np.degrees(lg.log[-1]["ang"])) if lg.log else 0.0
        close = True
        if ph == "approach":
            k = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2))
            s_ = float(np.clip((t - ts[k]) / max(1e-9, ts[k + 1] - ts[k]), 0, 1))
            q, g = Q[k] + s_ * (Q[k + 1] - Q[k]), G[k] + s_ * (G[k + 1] - G[k])
            close = modes[k + 1] == "close"
            if t >= ts[-1]:
                go("turn", t)
        elif ph in ("turn", "settle"):
            if ph == "turn":
                st["cmd"] = min(st["cmd"] + TURN_RATE * dt, ang + TURN_LEAD, T_turn["deg"][-1])
                if abs(ang - target_deg) <= TRIM_DEG:
                    go("settle", t)
            q, g = interp(T_turn, "deg", st["cmd"])
            if ph == "settle" and t - st["t_phase"] >= T_SETTLE:
                go("open", t)
        elif ph == "open":
            q, g = interp(T_turn, "deg", st["cmd"])
            close = False
            if t - st["t_phase"] >= T_OPEN:
                go("retreat", t)
        elif ph in ("retreat", "still") and RT is not None:
            tau_r = min(t - st["t_phase"], RT[-1]) if ph == "retreat" else RT[-1]
            k = int(np.clip(np.searchsorted(RT, tau_r, side="right") - 1, 0, len(RT) - 2))
            s_ = float(np.clip((tau_r - RT[k]) / max(1e-9, RT[k + 1] - RT[k]), 0, 1))
            q, g = RQ[k] + s_ * (RQ[k + 1] - RQ[k]), RG[k] + s_ * (RG[k + 1] - RG[k])
            close = False
            if ph == "retreat" and tau_r >= RT[-1]:
                go("still", t)
            if ph == "still" and t - st["t_phase"] >= T_STILL:
                raise StopRun
        elif ph in ("retreat", "still"):
            u = min(1.0, (t - st["t_phase"]) / T_RETREAT) if ph == "retreat" else 1.0
            q, g = interp(T_ret, "mm", u * T_ret["mm"][-1])
            close = False
            if ph == "retreat" and u >= 1.0:
                go("still", t)
            if ph == "still" and t - st["t_phase"] >= T_STILL:
                raise StopRun
        if t >= TIMEOUT_S:
            raise StopRun
        qt = rig.control.joint_target_q.numpy()
        for n, v in zip(arm_names, q):
            qt[tq_index[n]] = v
        for n in fingers:
            qt[tq_index[n]] = FINGER_CLOSE if close else FINGER_OPEN
        rig.control.joint_target_q.assign(qt)
        jf = rig.control.joint_f.numpy()
        jf[f_index] = g
        rig.control.joint_f.assign(jf)

    try:
        rec = rig.run(TIMEOUT_S + 1.0, control, lg.extra)
    except StopRun:
        rec = rig.rec
    print("phases:", st["phases"])
    paths = rec.save("rollout", force_keys=["knob_deg", "pads_N", "robot_object_contact_N"])
    print("saved:", paths)
    L = lg.log
    final = float(np.degrees(L[-1]["ang"]))
    out = {"final_deg": round(final, 2), "target_deg": target_deg, "error_deg": round(final - target_deg, 2),
           "phases": st["phases"], "robot": "Unitree G1 + Dex1", "robot_variant": True, "seed": a.seed}
    (rig.out_dir / "g1_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    wp.init()
    main()
