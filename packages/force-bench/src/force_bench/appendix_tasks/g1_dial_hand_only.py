"""Diagnosis: the Dex1 hand WITHOUT the arm, closing on the dial's bar (stage 1 of the dial-clamp study).

The hand's base is welded to the world at the pose a full G1 run had at its first frame (``--ref`` folder:
rollout_states.npz + g1_jaw_trace.jsonl from turn_dial_g1_expert with G1_DIAG_TRACE=1), and its finger joints are
driven along that run's own finger targets, frame by frame.  The dial is the same seed's.  Nothing else changes: the
same finger model (G1_FINGER_COLLISION), drive, effort (the run's squeeze), kd scale, solver and contacts.  Every
frame logs each finger contact (point, normal, normal and tangential force, separation, pair) and the count per jaw.

    python -m force_bench.appendix_tasks.g1_dial_hand_only --model g1_dex1_model.npz --seed 1 --ref DIR \\
        --out OUT --duration 1.0 [--squeeze 70 --kd-scale 4]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

import force_bench.appendix_tasks.turn_dial_to_target as T
from force_bench.appendix_tasks.common import appendix_rig, sample_start
from force_bench.appendix_tasks.install_light_bulb_g1 import FLOOR_CB, TABLE_HALF
from force_bench.g1 import robot as G1R
from force_bench.g1.robot import PREFIX, ContactPeak, MagnitudeProbe, add_g1, floor_and_table
from force_bench.kit import add_asset, place_on_table
from force_bench.rig import make_solver

DIAL, KNOB = "standard", "flat"
SIDE = "left"


def _xyzw_to_R(q):
    x, y, z, w = (float(v) for v in q)
    return G1R._rot((w, x, y, z))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ref", required=True, help="the full run's appendix folder (rollout_states.npz, g1_jaw_trace)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--squeeze", type=float, default=70.0, help="the finger effort cap the run's close set (N)")
    ap.add_argument("--kd-scale", type=float, default=float(os.environ.get("G1_FINGER_KD_SCALE", 1.0)))
    a = ap.parse_args(argv)
    ref = Path(a.ref)
    z = np.load(ref / "rollout_states.npz", allow_pickle=True)
    lab = [str(x) for x in z["body_labels"]]
    hb = lab.index(f"{PREFIX}{SIDE}_hand_base_link")
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = _xyzw_to_R(z["body_q"][0, hb, 3:7]), z["body_q"][0, hb, :3]
    trace = [json.loads(line) for line in open(ref / "g1_jaw_trace.jsonl")]
    tt = np.array([r["t"] for r in trace])
    tq = np.array([r["q_target_mm"] for r in trace]) * 1e-3

    start = sample_start({"dial": T.DIAL_PLACE}, a.seed)["dial"]
    key, choice = T.DIAL_KEYS[DIAL], {"Knob geometry": T.KNOB_LABELS[KNOB]}
    rig = appendix_rig("g1_dial_hand_only")
    rig.out_dir = Path(a.out)
    rig.out_dir.mkdir(parents=True, exist_ok=True)
    rig.substeps = T.SUBSTEPS
    rig.collide_every_substep = True
    g1 = add_g1(rig.builder, a.model, wp.transform_identity(), hand_only=(f"{SIDE}_hand_base_link", X))
    rig.robot_base = wp.transform_identity()
    stance = tuple(float(v) for v in os.environ.get("G1_DIAL_STANCE", "0.33,0.10").split(","))
    px = float(T.DIAL_XY[0] - stance[0])                     # the table as turn_dial_g1_expert places it
    tc = (px + 0.12 + TABLE_HALF[0], float(T.DIAL_XY[1]))
    rig.table(center_xy=tc, half=TABLE_HALF[0])
    floor_and_table(rig.builder, FLOOR_CB, tc, TABLE_HALF)
    pose = place_on_table(key, choice, xy=(start[0], start[1]), yaw=float(start[2]))
    h = add_asset(rig, key, choice, pose=pose)
    for i in range(len(rig.builder.shape_label)):
        bi = rig.builder.shape_body[i]
        if bi >= 0 and str(rig.builder.body_label[bi]).startswith(PREFIX):
            rig.builder.shape_gap[i] = T.ROBOT_GAP
    fingers = [f"{SIDE}_dex1_finger_joint_{k}" for k in (1, 2)]
    fdofs = [int(rig.builder.joint_qd_start[g1["joints"][n]]) for n in fingers]
    for d in fdofs:
        rig.builder.joint_target_kd[d] = rig.builder.joint_target_kd[d] * a.kd_scale
        rig.builder.joint_effort_limit[d] = a.squeeze
    q0 = tq[0]
    for n, v in zip(fingers, q0):
        qi = int(rig.builder.joint_q_start[g1["joints"][n]])
        rig.builder.joint_q[qi] = float(v)
        rig.builder.joint_target_q[qi] = float(v)
    shift = np.array([start[0] - T.DIAL_XY[0], start[1] - T.DIAL_XY[1], 0.0])
    rig.finalize(probes=[f"{h.label}/knob"], cam_pos=T.CAM_POS + shift, cam_target=T.CAM_TGT + shift)
    rig.solver = make_solver(rig.model, iterations=T.ITERS, impratio=T.IMPRATIO)
    labels = [str(x or "") for x in rig.model.shape_label]
    knob_sh = [i for i, l in enumerate(labels) if l.startswith(f"kit/{h.label}/") and "knob" in l]
    sb = rig.model.shape_body.numpy()
    blab = [str(x) for x in rig.model.body_label]
    jaws = {k: [i for i in range(rig.model.shape_count) if sb[i] >= 0 and blab[sb[i]] in
                (f"{PREFIX}{SIDE}_hand_Link{k}_2", f"{PREFIX}{SIDE}_hand_Link{k}_3")] for k in (1, 2)}
    for k in (1, 2):
        rig.probes[f"g1/finger{k}"] = MagnitudeProbe(rig.model, [f"{PREFIX}{SIDE}_hand_Link{k}_{j}" for j in (2, 3)],
                                                     counterpart_shapes=[labels[i] for i in knob_sh])
    rig.contact_peak = rig.probes["g1/contact"] = ContactPeak(rig.model)
    qs = rig.model.joint_q_start.numpy()
    fq = [int(qs[g1["joints"][n]]) for n in fingers]
    tqi = fq                                         # 1-dof joints: target index == q index
    log = []

    def control(t):
        qt = rig.control.joint_target_q.numpy()
        k = int(np.clip(np.searchsorted(tt, t), 0, len(tt) - 1))
        qt[tqi] = tq[k]
        rig.control.joint_target_q.assign(qt)

    def extra(state, t):
        q = state.joint_q.numpy()
        row = {"t": round(float(t), 4), "q_mm": [round(1e3 * float(q[i]), 3) for i in fq],
               "q_target_mm": [round(1e3 * float(v), 3) for v in rig.control.joint_target_q.numpy()[tqi]]}
        for k in (1, 2):
            pr = rig.probes[f"g1/finger{k}"].history
            row[f"jaw{k}_normal_N"] = round(float(pr[-1]["force"]), 3) if pr else 0.0
            row[f"jaw{k}_full_N"] = round(float(pr[-1]["force_full"]), 3) if pr else 0.0
            cs = G1R.contact_log(rig.model, rig.contacts, state, jaws[k], knob_sh)
            row[f"contacts_jaw{k}"] = cs
            row[f"count_jaw{k}"] = len(cs)
        log.append(row)
        return {}

    rig.run(a.duration, control, extra)
    (rig.out_dir / "hand_only_trace.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    nrm = np.array([[r["jaw1_normal_N"], r["jaw2_normal_N"]] for r in log])
    full = np.array([[r["jaw1_full_N"], r["jaw2_full_N"]] for r in log])
    over = nrm > 150.0
    def run_len(col):
        best = cur = 0
        for v in col:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best
    allc = [c for r in log for k in (1, 2) for c in r[f"contacts_jaw{k}"]]
    ratio = [float(np.linalg.norm(c["f_tan_N"])) / c["f_normal_N"] for c in allc if c["f_normal_N"] > 1.0]
    summary = {"finger_collision": G1R.FINGER_COLLISION, "ref": str(ref), "seed": a.seed, "squeeze_N": a.squeeze,
               "kd_scale": a.kd_scale, "hand_base_pose_from_ref_frame0": X.round(5).tolist(),
               "per_jaw_normal_peak_N": nrm.max(0).round(1).tolist(), "per_jaw_full_peak_N": full.max(0).round(1).tolist(),
               "frames_over_150N": over.sum(0).tolist(), "longest_run_over_150N": [run_len(over[:, 0]), run_len(over[:, 1])],
               "max_single_frame_contact": rig.contact_peak.peak, "max_single_frame_contact_normal": rig.contact_peak.peak_normal,
               "contact_tan_over_normal": {"n": len(ratio), "median": round(float(np.median(ratio)), 3) if ratio else None,
                                           "p95": round(float(np.percentile(ratio, 95)), 3) if ratio else None,
                                           "share_at_cone_edge_0.99": round(float(np.mean(np.array(ratio) > 0.99)), 3) if ratio else None},
               "count_per_jaw_max": [max(r["count_jaw1"] for r in log), max(r["count_jaw2"] for r in log)],
               "min_sep_mm": round(min((c["sep_mm"] for c in allc), default=0.0), 3)}
    (rig.out_dir / "hand_only_summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    wp.init()
    main()
