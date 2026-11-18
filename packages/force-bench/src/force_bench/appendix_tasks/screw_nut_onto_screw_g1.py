"""Appendix task "Screw nut onto screw" (5.7) on the Unitree G1 + Dex1 (an embodiment variant), in matrix's Isaac
setup.

The FR3 row (``screw_nut_onto_screw``: the printed bolt and free hex nut on the real SDF thread, the nut starting in
the closed pinch 0.6 turn onto the first thread, the suite's RegraspDriver, the row's verdict) with the G1 in the
FR3's place, as the G1 bulb row does it:

* the G1 stands FIXED relative to the table, the bolt STANCE ahead of / beside its pelvis (a top-down reach sweep at
  the nut's height); the bolt does not move (the FR3 row has no seeded start either);
* the unchanged RegraspDriver drives the Dex1's pad centre through ``G1IK`` in the FR3 TCP convention; its stroke
  window (+-YAW0 about YAW_BIAS) is the G1's own, measured here at the start and the engaged height, its twist and
  return held to the G1's wrist rule (<= 0.9 rad/s), and the stroke count scaled so the strokes cover what the FR3's
  seven 160-degree strokes do;
* each Dex1 jaw's two colliding links are one finger (finL / finR, the per-finger force the force limit reads);
* the arm's gravity torques are fed forward live, each actuator capped at the URDF limit less that budget.

Run: python -m force_bench.appendix_tasks.screw_nut_onto_screw_g1 --model g1_dex1_model.npz
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import trimesh
import warp as wp

import force_bench.appendix_tasks.screw_nut_onto_screw as SN
from force_bench.appendix_tasks.common import (RobotContactProbe, appendix_rig, check_released_and_clear, ease,
                                               save_appendix_verdict)
from force_bench.appendix_tasks.install_light_bulb_g1 import (FLOOR_CB, PELVIS_Z_CB, TABLE_HALF, WRIST_MAX, dex1_q,
                                                              wrist_window)
from force_bench.g1.robot import (PREFIX, ContactPeak, G1IK, SumProbe, TorqueMonitor, add_g1, apply_look,
                                  camera_mounts, cap_headroom, effective_arm_caps, floor_and_table, gravity_torques,
                                  init_pose, sim_cap_readback)
from force_bench.probes import ForceProbe
from force_bench.rig import make_solver
from force_bench.robot.franka import down_grasp_quat
from force_bench.tasks import nut_bolt_arm as NA
from force_bench.tasks.nut_bolt import (BOLT_WORLD_C, LEAD, MATE_YAW_DEG, MATE_ZC, SDF_MARGIN, SDF_NB, SDF_RES,
                                        load_subdivided)
from force_bench.tasks.nut_bolt_arm import PairProbe, RegraspDriver, build_plan
from force_bench.tasks.wrench_nut_arm import hex_flat_phase

NAME = "screw_nut_onto_screw_g1"
SIDE = "right"
STANCE = (0.30, -0.15)              # m: the bolt axis ahead of / to the left of the pelvis (set from the reach sweep)
FR3_SWEEP = 7 * math.radians(160.0)  # what the FR3 row's strokes cover


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Screw nut onto screw (in hand) on the G1 + Dex1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=None, help="accepted for the runner; the FR3 row has no seeded start")
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    bake_yaw = (MATE_YAW_DEG + (SN.Z_START - MATE_ZC) / LEAD * 360.0) % 360.0
    rig = appendix_rig(a.name or NAME)
    axis = np.array([0.0, 0.0])
    pelvis = wp.transform(wp.vec3(axis[0] - STANCE[0], axis[1] - STANCE[1], PELVIS_Z_CB), wp.quat_identity())
    g1 = add_g1(rig.builder, a.model, pelvis)
    rig.robot_base = pelvis
    px = float(axis[0] - STANCE[0])
    tc = (px + 0.12 + TABLE_HALF[0], float(axis[1]))
    rig.table(center_xy=tc, half=TABLE_HALF[0])
    floor_and_table(rig.builder, FLOOR_CB, tc, TABLE_HALF)

    from force_bench.assets import make_newton_mesh, set_body_mass
    bolt_tm = load_subdivided("fasteners", "screw_075mm_1_body1")
    bolt_mesh, bolt_c = make_newton_mesh(bolt_tm, sdf=True, sdf_resolution=SDF_RES, narrow_band=SDF_NB,
                                         margin=SDF_MARGIN)
    bolt_sid = rig.builder.add_shape_mesh(-1, mesh=bolt_mesh,
                                          xform=wp.transform(wp.vec3(*BOLT_WORLD_C), wp.quat_identity()),
                                          cfg=rig.cfg(mu=NA.NUT_MU, gap=NA.GAP), label="bolt")
    rig.shapes["bolt"] = [bolt_sid]
    bolt_world_tm = bolt_tm.copy()
    bolt_world_tm.apply_translation(BOLT_WORLD_C - np.asarray(bolt_c))
    rig.meshes["bolt"] = (bolt_world_tm, BOLT_WORLD_C)
    nut_tm = load_subdivided("fasteners", "nut_075mm_1_body1")
    nut_tm.apply_translation(-(nut_tm.bounds[0] + nut_tm.bounds[1]) / 2)
    nut_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(bake_yaw), [0, 0, 1]))
    flat = hex_flat_phase(nut_tm)
    nut_mesh, nut_c = make_newton_mesh(nut_tm, sdf=True, sdf_resolution=SDF_RES, narrow_band=SDF_NB,
                                       margin=SDF_MARGIN)
    nut_pos = np.array([0.0, 0.0, SN.Z_START])
    nut_body = rig.builder.add_body(xform=wp.transform(wp.vec3(*nut_pos), wp.quat_identity()), label="nut")
    nut_sid = rig.builder.add_shape_mesh(nut_body, mesh=nut_mesh, cfg=rig.cfg(mu=NA.NUT_MU, gap=NA.GAP),
                                         label="nut_shape")
    rig.shapes["nut"] = [nut_sid]
    set_body_mass(rig.builder, nut_body, 0.01)
    nut_world_tm = nut_tm.copy()
    nut_world_tm.apply_translation(nut_pos - np.asarray(nut_c))
    rig.meshes["nut"] = (nut_world_tm, nut_pos)
    for i in range(len(rig.builder.shape_label)):           # the default 0.1 m gap floods the thread meshes
        bi = rig.builder.shape_body[i]
        if bi >= 0 and str(rig.builder.body_label[bi]).startswith(PREFIX):
            rig.builder.shape_gap[i] = NA.GAP

    # ── the G1's arm: IK, the stroke window, the start pose, the torque budget ─────────────────────────────────
    start_q = init_pose(a.model)
    ik = G1IK(a.model, pelvis, SIDE, start_q)
    ik.arm_joints = [ik.g1["joints"][n] for n in ik.arm]
    tcp0 = np.array([0.0, 0.0, SN.Z_START + NA.TCP_OFF])
    yaw_hi, yaw_lo, q_hi, gmax = wrist_window(ik, axis, tcp0[2], tcp0[2] - SN.FLOOR_DROP)
    # the window in the driver's terms: strokes sweep +-YAW0 about YAW_BIAS, the twist and the return at the G1's
    # wrist rule, enough strokes to cover the FR3's sweep
    NA.YAW_BIAS = 0.5 * (yaw_hi + yaw_lo)
    NA.YAW0 = 0.5 * (yaw_hi - yaw_lo)
    NA.OMEGA = min(NA.OMEGA, WRIST_MAX)
    NA.OMEGA_SEAT = min(NA.OMEGA_SEAT, WRIST_MAX)
    NA.T_RET = max(NA.T_RET, 1.875 * 2 * NA.YAW0 / WRIST_MAX)
    t_stroke = max(NA.T_STROKE, 2 * NA.YAW0 / NA.OMEGA + 0.4)
    n_strokes = int(math.ceil(FR3_SWEEP / (2 * NA.YAW0)))
    NA.HAND_BODY = PREFIX + f"{SIDE}_hand_base_link"
    NA.FINGER_BODIES = (PREFIX + f"{SIDE}_hand_Link1_3", PREFIX + f"{SIDE}_hand_Link2_3")
    base = flat - np.pi / 2 - NA.YAW_BIAS                   # the pads on two flats (as the FR3 row picks it)
    k = np.round((NA.YAW0 - base) / (np.pi / 3))
    yaw_init = float(base + k * np.pi / 3)
    if yaw_init > NA.YAW0 + 0.35:
        yaw_init -= np.pi / 3
    ik.joint_q.assign(q_hi)
    q_sol = ik.solve(tcp0, down_grasp_quat(yaw_init + NA.YAW_BIAS), iterations=96)
    budget = gmax + 0.5
    rig.effort_budget = cap_headroom(rig.builder, g1, ik.arm, budget)
    qs_b = rig.builder.joint_q_start
    for n, v in start_q.items():
        if n in g1["joints"]:
            rig.builder.joint_q[int(qs_b[g1["joints"][n]])] = float(v)
            rig.builder.joint_target_q[int(qs_b[g1["joints"][n]])] = float(v)
    ik_qs = ik.model.joint_q_start.numpy()
    jmap = {n: g1["joints"][n] for n in ik.arm}
    for n in ik.arm:
        v = float(q_sol[int(ik_qs[ik.g1["joints"][n]])])
        rig.builder.joint_q[int(qs_b[jmap[n]])] = v
        rig.builder.joint_target_q[int(qs_b[jmap[n]])] = v
    fingers = [f"{SIDE}_dex1_finger_joint_1", f"{SIDE}_dex1_finger_joint_2"]
    for n in fingers:
        rig.builder.joint_q[int(qs_b[g1["joints"][n]])] = dex1_q(SN.FING_TOUCH)
        rig.builder.joint_target_q[int(qs_b[g1["joints"][n]])] = dex1_q(SN.FING_TOUCH)

    cam_c = np.array([0.0, 0.0, 0.022])
    links = {i: [f"{PREFIX}{SIDE}_hand_Link{i}_{j}" for j in (2, 3)] for i in (1, 2)}
    rig.finalize(probes=["nut"], cam_pos=cam_c + np.array([0.07, -0.10, 0.055]), cam_target=cam_c)
    rig.probes = {
        "seat": PairProbe(rig.model, "nut", counterpart_shapes=["bolt"]),
        # one finger = a Dex1 jaw's two colliding links, against the nut
        "finL": SumProbe([ForceProbe(rig.model, sensing_body=lb, counterpart_bodies=["nut"]) for lb in links[1]]),
        "finR": SumProbe([ForceProbe(rig.model, sensing_body=lb, counterpart_bodies=["nut"]) for lb in links[2]]),
        "robot_object": RobotContactProbe(rig.model, ["nut"], static_shapes=["bolt"]),
    }
    rig.pad_probes = ("finL", "finR")
    rig.contact_peak = rig.probes["g1/contact"] = ContactPeak(rig.model)
    rig.finger_bodies = NA.FINGER_BODIES
    rig.solver = make_solver(rig.model, iterations=20)
    rig.arm_caps_sim = sim_cap_readback(rig.solver, g1, rig.effort_budget, tag=f"[{NAME}]")
    rig.arm_caps = effective_arm_caps(rig.model, g1)
    apply_look(rig.model, g1)
    rig.camera_mounts = camera_mounts(a.model, SIDE)
    ni = rig.body_index("nut")
    wi = rig.body_index(NA.HAND_BODY)

    tqs = rig.model.joint_target_q_start.numpy()
    dqs = rig.model.joint_qd_start.numpy()
    arm_t = np.array([int(tqs[jmap[n]]) for n in ik.arm])
    arm_d = np.array([int(dqs[jmap[n]]) for n in ik.arm])
    arm_j = [jmap[n] for n in ik.arm]
    fin_t = [int(tqs[g1["joints"][n]]) for n in fingers]
    ik_arm_q = np.array([int(ik_qs[ik.g1["joints"][n]]) for n in ik.arm])
    last = {"fingers": SN.FING_TOUCH}

    def drive_arm(tcp_pos, tcp_quat, fingers=0.04):
        q = ik.solve(tcp_pos, tcp_quat, iterations=16)
        qt = rig.control.joint_target_q.numpy()
        qt[arm_t] = q[ik_arm_q]
        qt[fin_t] = dex1_q(float(fingers))
        last["fingers"] = float(fingers)
        rig.control.joint_target_q.assign(qt)
        jf = rig.control.joint_f.numpy()
        jf[arm_d] = np.clip(gravity_torques(rig.model, rig.s0.body_q.numpy(), arm_j), -budget, budget)
        rig.control.joint_f.assign(jf)

    rig.ik, rig.drive_arm = ik, drive_arm
    rig.torque_monitor = TorqueMonitor(rig.model, g1, ik.arm, {n: int(tqs[jmap[n]]) for n in ik.arm})
    rig.g1_plan = {"task": "nut", "seed": a.seed, "arm": SIDE, "fixed_stance": list(STANCE),
                   "pelvis_cb": [float(v) for v in wp.transform_get_translation(pelvis)], "door_shift_m": [0.0, 0.0],
                   "stroke_window_deg": [round(math.degrees(yaw_hi), 1), round(math.degrees(yaw_lo), 1)],
                   "strokes": n_strokes, "t_stroke_s": round(t_stroke, 2), "return_s": round(NA.T_RET, 2),
                   "twist_rad_s": NA.OMEGA, "orient": "top-down", "waist_pitch": None, "gamma_deg": None,
                   "stance_step_m": [0.0, 0.0]}
    (rig.out_dir / "g1_plan.json").write_text(json.dumps(rig.g1_plan, indent=1))
    print(f"[{NAME}] window {math.degrees(yaw_hi):+.0f}..{math.degrees(yaw_lo):+.0f} deg "
          f"({math.degrees(2 * NA.YAW0):.0f} per stroke), {n_strokes} strokes of {t_stroke:.2f} s, "
          f"return {NA.T_RET:.2f} s, yaw_init {math.degrees(yaw_init):+.0f}", flush=True)

    plan, t_plan = build_plan([(-1.0, n_strokes, False)], t_settle=0.0, t_stroke=t_stroke)
    drv = RegraspDriver(rig, plan, tcp0, ni, wi, part_z0=float(nut_pos[2]), yaw_init=yaw_init,
                        finger_grip=NA.FINGER_GRIP, latch_mm=99.0, floor_drop=SN.FLOOR_DROP, f_seat=NA.F_SEAT,
                        lead_m=LEAD, axis_xy=(0.0, 0.0), yaw_bias=NA.YAW_BIAS, f_grip_max=NA.F_GRIP_MAX,
                        f_grip_ok=NA.F_GRIP_OK, relax_open_max=NA.RELAX_OPEN_MAX,
                        relax_close_max=NA.RELAX_CLOSE_MAX, relax_rate=NA.RELAX_RATE, omega=NA.OMEGA,
                        yaw0=NA.YAW0)
    drv.ctl["fingers"] = SN.FING_TOUCH
    duration = t_plan + SN.T_RETREAT + SN.T_STILL
    end = {"p": None}

    def control(t):
        if t < t_plan:
            drv.control(t)
            return
        if end["p"] is None:
            tcp = rig.ik.tcp_pose()
            end["p"], end["q"] = tcp[:3].copy(), _fr3_quat(tcp[3:7])
            end["f"] = last["fingers"]
        u = ease((t - t_plan) / SN.T_RETREAT)
        rig.drive_arm(end["p"] + np.array([0.0, 0.0, SN.RETREAT * u]), end["q"], fingers=end["f"])

    log = []

    def extra(state, t):
        rig.torque_monitor.sample(state, rig.control, t, "plan" if t < t_plan else "retreat")
        out = drv.extra(state, t) if t < t_plan and "_ny" in drv.ctl else {}
        bq = state.body_q.numpy()
        rc = rig.probes["robot_object"].history[-1]["force"] if rig.probes["robot_object"].history else 0.0
        log.append((t, float(bq[ni, 2]), rc,
                    rig.probes["finL"].history[-1]["force"] if rig.probes["finL"].history else 0.0,
                    rig.probes["finR"].history[-1]["force"] if rig.probes["finR"].history else 0.0))
        out.update({"nut_z_mm": round(float(bq[ni, 2]) * 1000, 3), "robot_object_contact_N": round(rc, 4)})
        return out

    rec = rig.run(duration, control, extra)
    paths = rec.save("rollout", force_keys=["seat_N", "finL_N", "finR_N", "robot_object_contact_N"])
    print("saved:", paths)
    qa = rig.qa(paths)
    L = np.array(log)
    t_a, z_a, rc_a, fl, fr = (L[:, k] for k in range(5))
    turns = (SN.Z_START - float(z_a[t_a < t_plan].min())) / LEAD
    g0 = t_a < SN.T_GRIP0 + 0.3
    strokes_m = np.zeros_like(t_a, bool)
    for s in plan:
        if s["kind"] == "stroke":
            strokes_m |= (t_a >= s["t0"]) & (t_a < s["t1"])
    z_rel = float(np.median(z_a[(t_a >= t_plan - 0.3) & (t_a < t_plan)]))
    crit = {
        "held_in_hand_at_start": {"pass": bool(min(fl[g0].max(), fr[g0].max()) > 1.0
                                               and abs(z_a[g0].min() - SN.Z_START) < 0.001),
                                  "pad_peak_N": [round(float(fl[g0].max()), 2), round(float(fr[g0].max()), 2)]},
        "threads_fully_engaged": {"pass": bool(float(z_a[-1]) <= SN.ENGAGED_Z),
                                  "final_nut_centre_mm": round(float(z_a[-1]) * 1000, 2),
                                  "engaged_when_centre_below_mm": round(SN.ENGAGED_Z * 1000, 2),
                                  "turns_screwed": round(turns, 2)},
        "nut_stays_put_after_release": {"pass": bool(abs(float(z_a[-1]) - z_rel) < 0.0003),
                                        "z_before_release_mm": round(z_rel * 1000, 3),
                                        "final_z_mm": round(float(z_a[-1]) * 1000, 3)},
        "real_finger_grasp": {"pass": bool(fl[strokes_m].mean() > 1.0 and fr[strokes_m].mean() > 1.0),
                              "mean_pad_N_in_strokes": [round(float(fl[strokes_m].mean()), 2),
                                                        round(float(fr[strokes_m].mean()), 2)]},
    }
    check_released_and_clear(qa, t_a, rc_a, t_plan + SN.T_RETREAT)
    crit["released_and_clear"] = qa.report["checks"]["released_and_clear"]
    for k_, c in crit.items():
        qa.report["checks"].setdefault(k_, c)
    qa.report["pass"] = all(c.get("pass", True) for c in qa.report["checks"].values())
    report = rig.finish(qa)
    save_appendix_verdict(rig, task="Screw nut onto screw (G1 + Dex1)",
                          source="force_bench.appendix_tasks.screw_nut_onto_screw with the G1 in the FR3's place",
                          spec="The robot movement stops after terminating the rotation motion, moving away from, "
                               "and breaking contact with the task objects.",
                          criteria=crit, physics_qa=report,
                          changes=["the G1 + Dex1 (fixed stance, the right arm) instead of the FR3",
                                   f"{n_strokes} strokes over the G1's {math.degrees(2 * NA.YAW0):.0f} deg window "
                                   "at <= 0.9 rad/s"])


def _fr3_quat(hand_quat_xyzw):
    """The hand's orientation back in the FR3 TCP convention drive_arm takes."""
    from force_bench.g1.robot import TCP_TO_HAND, _R_xyzw
    from force_bench.policy_env.cameras import _quat_xyzw
    return np.asarray(_quat_xyzw(_R_xyzw(hand_quat_xyzw) @ TCP_TO_HAND.T), float)


if __name__ == "__main__":
    wp.init()
    main()
