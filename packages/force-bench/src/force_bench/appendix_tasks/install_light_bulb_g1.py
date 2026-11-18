"""Appendix task "Install light bulb" (5.2) on the Unitree G1 + Dex1 (an embodiment variant), in matrix's Isaac
setup (pelvis 0.793 m on straight legs, the table top 7.7 cm above the pelvis).

The kit's bulb row (``_kit_thread``: the kit bulb and socket, their thread physics, the FR3's screw expert and its
verdict), with the G1 in the FR3's place:

* the G1 stands FIXED relative to the table, the socket's nominal axis STANCE ahead of / beside its pelvis (an
  Isaac-setup reach sweep: a top-down Dex1 grasp turns 285 deg about the vertical there, the most of any stance);
  ``--seed N`` moves only the socket (the row's 0.15 m square), as on the FR3;
* the bulb starts in the right hand's pinch above the socket (the row's in-hand start);
* the unchanged screw expert drives the Dex1's pad centre through a Newton IK of the right arm (``G1IK``) in the
  FR3's TCP convention; its wrist window per stroke is the G1's own, measured here at build (top-down, at the start
  and the seated height), so the 720 deg take as many strokes as that window needs (3 at ~240 deg);
* the arm's gravity torques are fed forward live; each arm actuator is capped at the G1's URDF limit less the
  feed-forward budget, so actuator + feed-forward stays within the limit.

Run: python -m force_bench.appendix_tasks.install_light_bulb_g1 --model g1_dex1_model.npz [--seed N]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import warp as wp

import force_bench.appendix_tasks._kit_thread as kt
from force_bench.appendix_tasks.common import RobotContactProbe, appendix_rig, sample_start
from force_bench.g1.robot import (DEX1_HALF_GAP0, PREFIX, ContactPeak, G1IK, TorqueMonitor, add_g1, apply_look,
                                  camera_mounts, cap_headroom, effective_arm_caps, floor_and_table,
                                  gravity_torques, init_pose, sim_cap_readback)
from force_bench.robot.franka import down_grasp_quat

NAME = "install_light_bulb_g1"
SIDE = "right"
STANCE = (0.35, -0.15)              # m: the socket's nominal axis ahead of / to the left of the pelvis
PELVIS_Z_CB = 0.793 - 0.8703        # the pelvis below the table top (Isaac setup)
FLOOR_CB = -0.8703
TABLE_HALF = (0.30, 0.40)
YAW_STEP = math.radians(5.0)
YAW_MARGIN = math.radians(25.0)     # inside the measured window at both ends (at 10 deg the pads' release at the
#                                     window's low end let the arm jump 16 mm off the bulb's axis)
YAW_SPAN_MAX = math.radians(300.0)
IK_TOL = (0.001, math.radians(1.0))
WRIST_MAX = 0.9                     # rad/s: our G1 scripted policies' wrist rule (<= 1 rad/s), 10 % inside it
FF_MARGIN = 0.5                     # N.m of feed-forward budget above the largest measured over the window
DEX1_Q = (-0.02, 0.0245)            # the finger joints' range


def dex1_q(fr3_dof: float) -> float:
    """The kit expert's FR3 finger target (pad face = dof - PAD_OFF from the centre) as the Dex1 finger joint that
    puts the pad face at the same half-gap."""
    return float(np.clip(DEX1_HALF_GAP0 - (fr3_dof - kt.PAD_OFF), *DEX1_Q))


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Install light bulb (in hand) on the G1 + Dex1")
    ap.add_argument("--model", required=True, help="the exported G1 + Dex1 model (g1_dex1_model.npz)")
    ap.add_argument("--bulb", dest="part", default=kt.TASKS["bulb"].defaults["part"],
                    choices=["lightbulb-1", "lightbulb-2", "lightbulb-3"])
    ap.add_argument("--size", default=kt.TASKS["bulb"].defaults["size"])
    ap.add_argument("--rev", default=kt.TASKS["bulb"].defaults["rev"], choices=["2", "2.5", "3", "3.5", "4"])
    ap.add_argument("--yaw0-deg", type=float, default=kt.YAW0_DEG)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def wrist_window(ik: G1IK, axis_xy, z_start, z_seat):
    """The G1's longest contiguous clockwise turn about the vertical, top-down, reachable at both heights from one
    IK branch: (yaw_hi, yaw_lo, the arm q at yaw_hi, the largest |gravity torque| per arm joint over it)."""
    q_home = ik.joint_q.numpy().copy()
    best = None
    for start in np.radians(np.arange(180.0, -181.0, -20.0)):
        ik.joint_q.assign(q_home)
        p0 = np.array([axis_xy[0], axis_xy[1], z_start])
        ik.solve(p0, down_grasp_quat(start), iterations=96)
        if max(a / b for a, b in zip(ik.residual(p0, down_grasp_quat(start)), IK_TOL)) > 1.0:
            continue
        q_start = ik.joint_q.numpy().copy()
        span, gmax = 0.0, np.zeros(len(ik.arm))
        while span < YAW_SPAN_MAX:
            y = start - span - YAW_STEP
            ok = True
            for z in (z_seat, z_start):                   # the seated height, then back up for the next step
                p = np.array([axis_xy[0], axis_xy[1], z])
                ik.solve(p, down_grasp_quat(y), iterations=24)
                if max(a / b for a, b in zip(ik.residual(p, down_grasp_quat(y)), IK_TOL)) > 1.0:
                    ok = False
                    break
                st = ik._state()
                gmax = np.maximum(gmax, np.abs(gravity_torques(ik.model, st.body_q.numpy(), ik.arm_joints)))
            if not ok:
                break
            span += YAW_STEP
        if best is None or span > best[0]:
            best = (span, start, q_start, gmax)
    if best is None or best[0] <= 2 * YAW_MARGIN:
        raise SystemExit(f"{NAME}: no top-down wrist window reaches the socket at {tuple(axis_xy)}")
    span, start, q_start, gmax = best
    return start - YAW_MARGIN, start - span + YAW_MARGIN, q_start, gmax


class G1ScrewExpert(kt.ScrewExpert):
    """The kit's screw expert, unchanged, with the arm torques sampled each frame."""

    def extra(self, state, t):
        self.rig.torque_monitor.sample(state, self.rig.control, t, self.st["phase"])
        return super().extra(state, t)


def setup(a, name):
    P = kt.measure_pair("bulb", a.part, a.size, a.rev)
    F = kt._fingers(P)
    rig = appendix_rig(name)
    rig.substeps = kt.SUBSTEPS
    rig.collide_every_substep = True
    nominal = np.asarray(kt.AXIS_XY, float)
    pelvis = wp.transform(wp.vec3(nominal[0] - STANCE[0], nominal[1] - STANCE[1], PELVIS_Z_CB), wp.quat_identity())
    g1 = add_g1(rig.builder, a.model, pelvis)
    rig.robot_base = pelvis
    px = float(nominal[0] - STANCE[0])
    floor_and_table(rig.builder, FLOOR_CB, (px + 0.12 + TABLE_HALF[0], float(nominal[1])), TABLE_HALF)
    for i in range(len(rig.builder.shape_label)):           # newton's 0.1 m default gap floods the thread meshes
        bi = rig.builder.shape_body[i]
        lbl = str(rig.builder.shape_label[i] or "")
        if (bi >= 0 and str(rig.builder.body_label[bi]).startswith(PREFIX)) or "ground" in lbl:
            rig.builder.shape_gap[i] = kt.ROBOT_GAP
    axis_xy = np.array(sample_start({"fixture": kt.AXIS_PLACE}, a.seed)["fixture"][:2])
    S = kt.build_scene(rig, P, "hand", np.radians(a.yaw0_deg), axis_xy)
    S.seed = a.seed
    tcp0 = S.p0 + kt.rotm(np.r_[S.p0, S.q0]) @ np.array([0.0, 0.0, P.grip_z])
    seat_tcp_z = S.fixture_world[2, 3] + P.seat_lift + P.grip_z
    # ── the G1's arm: IK, wrist window, start pose, torque budget ──────────────────────────────────────────────
    start_q = init_pose(a.model)
    ik = G1IK(a.model, pelvis, SIDE, start_q)
    jmap = {n: g1["joints"][n] for n in ik.arm}
    ik.arm_joints = [ik.g1["joints"][n] for n in ik.arm]
    yaw_hi, yaw_lo, q_hi, gmax = wrist_window(ik, S.axis_xy, tcp0[2], seat_tcp_z)
    kt.YAW_HI, kt.YAW_LO = yaw_hi, yaw_lo
    kt.HAND_BODY = PREFIX + f"{SIDE}_hand_base_link"
    # the G1's speed rule on the expert's FR3 timings: twist <= WRIST_MAX, and the open hand's turn back over the
    # window (a quintic: peak 1.875 x the mean rate) timed to the same peak.  At the FR3's 1.3 s the return peaked
    # ~2.5 rad/s and the lagging arm swung the pad centre 17 mm off the axis before the regrasp.
    for k in ("OMEGA", "OMEGA_ENGAGE", "OMEGA_TAKEUP", "OMEGA_SEAT"):
        setattr(kt, k, min(getattr(kt, k), WRIST_MAX))
    kt.T_RET = max(kt.T_RET, 1.875 * (yaw_hi - yaw_lo) / WRIST_MAX)
    ik.joint_q.assign(q_hi)
    q_sol = ik.solve(tcp0, down_grasp_quat(yaw_hi), iterations=96)
    budget = gmax + FF_MARGIN
    rig.effort_budget = cap_headroom(rig.builder, g1, ik.arm, budget)
    qs_b = rig.builder.joint_q_start
    for n, v in start_q.items():                            # both arms in the reset pose ...
        if n in g1["joints"]:
            rig.builder.joint_q[int(qs_b[g1["joints"][n]])] = float(v)
            rig.builder.joint_target_q[int(qs_b[g1["joints"][n]])] = float(v)
    ik_qs = ik.model.joint_q_start.numpy()
    for n in ik.arm:                                        # ... the active one at the start grasp
        v = float(q_sol[int(ik_qs[ik.g1["joints"][n]])])
        rig.builder.joint_q[int(qs_b[jmap[n]])] = v
        rig.builder.joint_target_q[int(qs_b[jmap[n]])] = v
    fingers = [f"{SIDE}_dex1_finger_joint_1", f"{SIDE}_dex1_finger_joint_2"]
    for n in fingers:
        rig.builder.joint_q[int(qs_b[g1["joints"][n]])] = dex1_q(F["touch"] - 0.0001)
        rig.builder.joint_target_q[int(qs_b[g1["joints"][n]])] = dex1_q(F["grip"])
    cam_pos, cam_tgt = kt.CAM["bulb"]["hand"]
    shift = np.r_[S.axis_xy - kt.AXIS_XY, 0.0]
    rig.finalize(probes=[P.kind], cam_pos=cam_pos + shift, cam_target=cam_tgt + shift)
    rig.solver = kt.make_solver(rig.model, iterations=kt.ITERS, impratio=kt.IMPRATIO)
    rig.arm_caps_sim = sim_cap_readback(rig.solver, g1, rig.effort_budget, tag=f"[{NAME}]")
    sc = rig.model.shape_color.numpy()
    for lbl, col in kt.VIS_COL.items():
        for sid in rig.shapes.get(lbl, []):
            sc[sid] = col
    rig.model.shape_color.assign(sc)
    apply_look(rig.model, g1)
    rig.arm_caps = effective_arm_caps(rig.model, g1)       # read back; refuses an uncapped arm
    sb = rig.model.shape_body.numpy()
    fin = {k: [int(s) for b in (f"{SIDE}_hand_Link{i}_2", f"{SIDE}_hand_Link{i}_3")
               for s in np.flatnonzero(sb == g1["bodies"][b])] for k, i in (("finL", 1), ("finR", 2))}
    seat_z_w = S.fixture_world[2, 3] + P.fixture_seat_z
    rig.probes = {"tap": kt.ContactTap(rig.model, S.part, S.part_shapes, S.fixture_shapes, fin, seat_z_w, P.seat_r,
                                       S.axis_xy),
                  "robot_object": RobotContactProbe(rig.model, [P.kind], static_shapes=S.static_labels)}
    rig.contact_peak = rig.probes["g1/contact"] = ContactPeak(rig.model)
    cj = int(np.flatnonzero(rig.model.joint_child.numpy() == S.part)[0])
    rig.solver = kt.ProbeTorque(rig.solver, int(rig.model.joint_qd_start.numpy()[cj]) + 5)
    rig.camera_mounts = camera_mounts(a.model, SIDE)
    # ── the drive the screw expert calls ───────────────────────────────────────────────────────────────────────
    tqs = rig.model.joint_target_q_start.numpy()
    dqs = rig.model.joint_qd_start.numpy()
    arm_t = np.array([int(tqs[jmap[n]]) for n in ik.arm])
    arm_d = np.array([int(dqs[jmap[n]]) for n in ik.arm])
    arm_j = [jmap[n] for n in ik.arm]
    fin_t = [int(tqs[g1["joints"][n]]) for n in fingers]
    ik_arm_q = np.array([int(ik_qs[ik.g1["joints"][n]]) for n in ik.arm])

    def drive_arm(tcp_pos, tcp_quat, fingers=0.04):
        q = ik.solve(tcp_pos, tcp_quat, iterations=16)
        qt = rig.control.joint_target_q.numpy()
        qt[arm_t] = q[ik_arm_q]
        qt[fin_t] = dex1_q(float(fingers))
        rig.control.joint_target_q.assign(qt)
        jf = rig.control.joint_f.numpy()
        g = gravity_torques(rig.model, rig.s0.body_q.numpy(), arm_j)
        jf[arm_d] = np.clip(g, -budget, budget)              # within the budget the actuator caps leave
        rig.control.joint_f.assign(jf)

    rig.ik, rig.drive_arm = ik, drive_arm
    tq_index = {n: int(tqs[jmap[n]]) for n in ik.arm}
    rig.torque_monitor = TorqueMonitor(rig.model, g1, ik.arm, tq_index)
    rig.g1_plan = {"task": "bulb", "seed": a.seed, "arm": SIDE, "fixed_stance": list(STANCE),
                   "pelvis_cb": [float(v) for v in wp.transform_get_translation(pelvis)],
                   "door_shift_m": [float(v) for v in (S.axis_xy - kt.AXIS_XY)],
                   "wrist_window_deg": [round(math.degrees(yaw_hi), 1), round(math.degrees(yaw_lo), 1)],
                   "twist_rad_s": kt.OMEGA, "return_s": round(kt.T_RET, 2),
                   "orient": "top-down", "waist_pitch": None, "gamma_deg": None, "stance_step_m": [0.0, 0.0]}
    print(f"[{NAME}] seed {a.seed}: socket shift {np.round((S.axis_xy - kt.AXIS_XY) * 1e3, 1).tolist()} mm, "
          f"wrist window {math.degrees(yaw_hi):+.0f}..{math.degrees(yaw_lo):+.0f} deg "
          f"({math.degrees(yaw_hi - yaw_lo):.0f} per stroke), ff budget {np.round(budget, 2).tolist()} N.m",
          flush=True)
    (rig.out_dir / "g1_plan.json").write_text(json.dumps(rig.g1_plan, indent=1))
    return rig, S, P, F, tcp0


def main(argv=None):
    a = parse(argv)
    name = a.name or NAME
    rig, S, P, F, tcp0 = setup(a, name)
    kt.preroll(rig, S, F, tcp0, "bulb")
    ex = G1ScrewExpert(rig, S, "hand", tcp0, kt.YAW_HI, F)
    try:
        rig.run(min(150.0, float(os.environ.get("G1_MAX_T", 150.0))), ex.control, ex.extra)   # G1_MAX_T: pre-flight
    except kt.StopRun:
        print(f"[{NAME}] rollout stopped at t={len(rig.rec.times) / rig.fps:.2f}s")
    paths = rig.rec.save("rollout", force_keys=["thread_N", "seat_N", "pad_torque_mNm", "robot_object_contact_N",
                                                "probe_mNm"])
    print("saved:", paths)
    kt.verdict(rig, S, ex, "hand", paths, kt.TASKS["bulb"])


if __name__ == "__main__":
    wp.init()
    main()
