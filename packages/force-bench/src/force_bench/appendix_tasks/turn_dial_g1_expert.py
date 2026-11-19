"""Appendix task "Turn dial to target" (5.6, the standard dial, flat knob, +90 deg) on the Unitree G1 + Dex1 (an
embodiment variant), in matrix's Isaac setup, driven by the FR3 row's own expert.

The FR3 row (``turn_dial_to_target``: the kit dial, its brake and detents, the FR3's Expert and its verdict) with the
G1 in the FR3's place, as the G1 bulb and nut rows do it:

* the G1 stands FIXED relative to the table, the dial's nominal axis STANCE ahead of / beside its pelvis; ``--seed``
  moves only the dial (the row's 0.15 m square and +-15 deg);
* the unchanged FR3 Expert (top-down, xy servo on the measured finger midpoint, close to a set squeeze, the turn
  closed on the measured dial angle, trim, open, retreat) drives the Dex1's pad centre through ``G1IK`` in the FR3
  TCP convention.  Its squeeze is the finger actuators' force cap (the FR3 row's 70 N for the flat knob, inside the
  Dex1's 5-120 N), on the row's own finger stiffness (20 kN/m), both read back from the simulator and recorded;
* the turn (target + the trim's clamp) must fit the G1's top-down wrist window, measured at build; the bar is
  symmetric under 180 deg, so the grasp may come from either side.  A seed where neither fits is a reach miss: the
  start frame is rendered and the miss reported;
* the arm's gravity torques are fed forward live, each actuator capped at the URDF limit less that budget.

Run: python -m force_bench.appendix_tasks.turn_dial_g1_expert --model g1_dex1_model.npz [--seed N]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import warp as wp

import newton

import force_bench.appendix_tasks.turn_dial_to_target as T
from force_bench.appendix_tasks.common import RobotContactProbe, StopRun, appendix_rig, sample_start
from force_bench.appendix_tasks.install_light_bulb_g1 import (FLOOR_CB, PELVIS_Z_CB, TABLE_HALF, WRIST_MAX, dex1_q,
                                                              render_start)
from force_bench.g1.robot import (PREFIX, ContactPeak, G1IK, SumProbe, TorqueMonitor, add_g1, apply_look,
                                  camera_mounts, cap_headroom, effective_arm_caps, finger_readback, floor_and_table,
                                  gravity_torques, init_pose, sim_cap_readback)
from force_bench.kit import add_asset, load_kit, place_on_table
from force_bench.rig import make_solver
from force_bench.robot.franka import down_grasp_quat
from force_bench.tasks.nut_bolt_arm import PairProbe

NAME = "turn_dial_g1_expert"
SIDE = os.environ.get("G1_DIAL_SIDE", "left")   # the left arm: its top-down wrist window lines up with the bar's
#                                                yaw at 6/10 seeds, the right arm's at 1/10 (g1_dial_stance_search.py)
STANCE = tuple(float(v) for v in os.environ.get("G1_DIAL_STANCE", "0.27,0.25").split(","))   # m, the dial axis
#                                                ahead of / to the left of the pelvis (the search's choice, left arm)
DIAL, KNOB, TARGET_DEG = "standard", "flat", 90.0
DEX1_TIP_PAST_PAD = 0.0273           # top-down: the fingertips this far below the pad centre (the FR3's are 9.93 mm)
TURN_NEED = math.radians(TARGET_DEG) + T.EXPERT_TRIM_CLAMP + math.radians(5.0)
MARGIN = math.radians(10.0)          # IK checked this far past both ends of the turn
assert T.OMEGA <= WRIST_MAX   # the turn, its trim and a margin
LAST_MISS: dict = {}


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Turn dial to target on the G1 + Dex1 (the FR3 expert)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def turn_fit(ik, axis, z, yaw0, gravity_joints):
    """The grasp yaw for the dial's turn: for the bar's yaw and its 180-deg twin (+ 2 pi k, kept within +-pi of 0),
    walk the top-down IK from the turn's top (a + TURN_NEED + MARGIN) down to its start (a - MARGIN) in 5-deg steps
    at the grasp height.  Returns (a, q at a, the largest |gravity torque| per arm joint over the walk) or None."""
    from force_bench.appendix_tasks.install_light_bulb_g1 import IK_TOL
    q_home = ik.joint_q.numpy().copy()
    p = np.array([axis[0], axis[1], z])
    for base in (yaw0, yaw0 + math.pi):
        a = math.remainder(base, 2 * math.pi)
        ys = np.arange(a + TURN_NEED + MARGIN, a - MARGIN - 1e-9, -math.radians(5.0))
        ik.joint_q.assign(q_home)
        ik.solve(p, down_grasp_quat(ys[0]), iterations=96)
        gmax, ok = np.zeros(len(ik.arm)), True
        for y in ys:
            ik.solve(p, down_grasp_quat(y), iterations=24)
            if max(e / tol for e, tol in zip(ik.residual(p, down_grasp_quat(y)), IK_TOL)) > 1.0:
                ok = False
                break
            gmax = np.maximum(gmax, np.abs(gravity_torques(ik.model, ik._state().body_q.numpy(), gravity_joints)))
        if ok:
            ik.solve(p, down_grasp_quat(a), iterations=48)
            return a, ik.joint_q.numpy().copy(), gmax
    return None


class G1DialExpert(T.Expert):
    """The FR3 row's Expert, unchanged, with the squeeze written to the Dex1 finger actuators and the arm torques
    sampled each frame."""

    finger_dofs: list = []

    def squeeze(self, f):
        if self._squeeze == f:
            return
        self._squeeze = f
        lim = self.rig.model.joint_effort_limit.numpy()
        lim[self.finger_dofs] = f
        self.rig.model.joint_effort_limit.assign(lim)
        self.rig.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def extra(self, state, t):
        self.rig.torque_monitor.sample(state, self.rig.control, t, self.st["phase"])
        return super().extra(state, t)


def main(argv=None):
    a = parse(argv)
    start = sample_start({"dial": T.DIAL_PLACE}, a.seed)["dial"]
    yaw0 = float(start[2])
    key, choice = T.DIAL_KEYS[DIAL], {"Knob geometry": T.KNOB_LABELS[KNOB]}
    rig = appendix_rig(a.name or NAME)
    rig.substeps = T.SUBSTEPS
    rig.collide_every_substep = True
    nominal = np.asarray(T.DIAL_XY, float)
    pelvis = wp.transform(wp.vec3(nominal[0] - STANCE[0], nominal[1] - STANCE[1], PELVIS_Z_CB), wp.quat_identity())
    g1 = add_g1(rig.builder, a.model, pelvis)
    rig.robot_base = pelvis
    px = float(nominal[0] - STANCE[0])
    tc = (px + 0.12 + TABLE_HALF[0], float(nominal[1]))
    rig.table(center_xy=tc, half=TABLE_HALF[0])
    floor_and_table(rig.builder, FLOOR_CB, tc, TABLE_HALF)
    pose = place_on_table(key, choice, xy=(start[0], start[1]), yaw=yaw0)
    h = add_asset(rig, key, choice, pose=pose)
    for i in range(len(rig.builder.shape_label)):           # newton's 0.1 m default gap floods the kit meshes
        bi = rig.builder.shape_body[i]
        if bi >= 0 and str(rig.builder.body_label[bi]).startswith(PREFIX):
            rig.builder.shape_gap[i] = T.ROBOT_GAP
    fingers = [f"{SIDE}_dex1_finger_joint_1", f"{SIDE}_dex1_finger_joint_2"]
    fdofs = [int(rig.builder.joint_qd_start[g1["joints"][n]]) for n in fingers]
    for d in fdofs:                                          # the row's own finger drive (the FR3 row sets its own)
        rig.builder.joint_target_ke[d] = T.FINGER_KE
        rig.builder.joint_target_kd[d] = T.FINGER_KD
    g = T._grip_geometry(load_kit()[key], choice, KNOB, h.info.get("up_gap_mm", 0.0) * T.MM)
    g = {**g, "tcp_z": g["tip_z"] + DEX1_TIP_PAST_PAD}       # the Dex1 pad centre above its fingertips
    axis = np.asarray(h.pose[:2, 3], float)
    tcp0 = np.array([axis[0], axis[1], g["tcp_z"]])

    # ── the G1's arm: the window, the grasp side, the start pose, the torque budget ───────────────────────────
    start_q = init_pose(a.model)
    ik = G1IK(a.model, pelvis, SIDE, start_q)
    ik.arm_joints = [ik.g1["joints"][n] for n in ik.arm]
    jmap = {n: g1["joints"][n] for n in ik.arm}
    fit = turn_fit(ik, axis, tcp0[2], yaw0, ik.arm_joints)
    miss = None
    if fit is None:
        LAST_MISS.update(dial_xy_m=[round(float(v), 4) for v in axis], dial_yaw_deg=round(math.degrees(yaw0), 1),
                         needed_deg=round(math.degrees(TURN_NEED + 2 * MARGIN), 1))
        miss = SystemExit(f"{NAME}: no top-down wrist window reaches the dial's turn: neither from the bar's yaw "
                          f"({math.degrees(yaw0):+.1f}) nor its 180-deg twin does the IK follow a "
                          f"{math.degrees(TURN_NEED + 2 * MARGIN):.0f} deg top-down turn at the grip")
        gmax = np.zeros(len(ik.arm))
    else:
        a0, q_a0, gmax = fit
    budget = gmax + 0.5
    rig.effort_budget = cap_headroom(rig.builder, g1, ik.arm, budget)
    qs_b = rig.builder.joint_q_start
    for n, v in start_q.items():
        if n in g1["joints"]:
            rig.builder.joint_q[int(qs_b[g1["joints"][n]])] = float(v)
            rig.builder.joint_target_q[int(qs_b[g1["joints"][n]])] = float(v)
    ik_qs = ik.model.joint_q_start.numpy()
    if miss is None:                                         # the arm at the grasp yaw, on the walked branch
        for n in ik.arm:
            v = float(q_a0[0][int(ik_qs[ik.g1["joints"][n]])] if q_a0.ndim == 2 else q_a0[int(ik_qs[ik.g1["joints"][n]])])
            rig.builder.joint_q[int(qs_b[jmap[n]])] = v
            rig.builder.joint_target_q[int(qs_b[jmap[n]])] = v
    f_open = g["half"] + T.PAD_OFF + T.OPEN_CLEAR
    for n in fingers:
        rig.builder.joint_q[int(qs_b[g1["joints"][n]])] = dex1_q(f_open)
        rig.builder.joint_target_q[int(qs_b[g1["joints"][n]])] = dex1_q(f_open)
    shift = np.array([start[0] - T.DIAL_XY[0], start[1] - T.DIAL_XY[1], 0.0])
    rig.finalize(probes=[f"{h.label}/knob"], cam_pos=T.CAM_POS + shift, cam_target=T.CAM_TGT + shift)
    rig.solver = make_solver(rig.model, iterations=T.ITERS, impratio=T.IMPRATIO)
    rig.arm_caps_sim = sim_cap_readback(rig.solver, g1, rig.effort_budget, tag=f"[{NAME}]")
    rig.arm_caps = effective_arm_caps(rig.model, g1)
    labels = [str(lbl) for lbl in rig.model.shape_label]
    knob_shapes = [lbl for lbl in labels if lbl.startswith(f"kit/{h.label}/") and "knob" in lbl]
    static_labels = [lbl for lbl in labels if lbl.startswith(f"kit/{h.label}/")
                     and any(s in lbl for s in ("_top", "_bottom", "gate_"))]
    moving = [rig.model.body_label[b] for b in h.bodies.values()]
    links = {i: [f"{PREFIX}{SIDE}_hand_Link{i}_{k}" for k in (2, 3)] for i in (1, 2)}
    probes = {"finL": SumProbe([PairProbe(rig.model, lb, counterpart_shapes=knob_shapes) for lb in links[1]]),
              "finR": SumProbe([PairProbe(rig.model, lb, counterpart_shapes=knob_shapes) for lb in links[2]]),
              "robot_object": RobotContactProbe(rig.model, moving, static_shapes=static_labels)}
    rig.probes.update(probes)
    rig.pad_probes = ("finL", "finR")
    rig.contact_peak = rig.probes["g1/contact"] = ContactPeak(rig.model)
    rig.camera_mounts = camera_mounts(a.model, SIDE)
    apply_look(rig.model, g1)
    rig.g1_plan = {"task": "dial", "seed": a.seed, "arm": SIDE, "fixed_stance": list(STANCE),
                   "pelvis_cb": [float(v) for v in wp.transform_get_translation(pelvis)],
                   "door_shift_m": [float(v) for v in (np.asarray(start[:2]) - nominal)], "dial_yaw_rad": yaw0,
                   "orient": "top-down", "waist_pitch": None, "gamma_deg": None, "stance_step_m": [0.0, 0.0]}
    if miss is not None:                                     # the start frame, then the miss
        rig.start_videos = render_start(rig)
        raise miss
    rig.g1_plan.update(turn_window_deg=[round(math.degrees(a0 - MARGIN), 1),
                                        round(math.degrees(a0 + TURN_NEED + MARGIN), 1)],
                       grasp_yaw_deg=round(math.degrees(a0), 1),
                       grasp_side="near" if abs(math.remainder(a0 - yaw0, 2 * math.pi)) < 1 else "far (180-deg twin)")
    rig.finger_params = finger_readback(rig.solver, rig.model, g1, SIDE,
                                        overdrive_m=1.5 * T.SQUEEZE[KNOB] / T.FINGER_KE)
    rig.variant_extra = {
        "dial_finger_squeeze_N": T.SQUEEZE[KNOB],
        "dial_finger_squeeze_source": "the FR3 dial row's flat-knob squeeze, set as each Dex1 finger actuator's force "
                                      "cap (Expert.squeeze; G1_FINGER_EFFORT=70 at build), inside the Dex1's 5-120 N; "
                                      "read back from the compiled model in finger_actuators_sim_readback",
        "dial_finger_stiffness_N_per_m": T.FINGER_KE,
        "dial_finger_stiffness_source": "the FR3 dial row's own finger position-drive gain (a controller gain, not a "
                                        "material property): the row closes 1.5 x squeeze / gain past contact",
        "dex1_tcp_above_tips_m": DEX1_TIP_PAST_PAD,
        "dex1_tcp_source": "the Dex1 pad centre is 27.3 mm above its fingertips (the FR3's TCP is 9.93 mm): the "
                           "expert's TCP height is the grip's tip height + 27.3 mm",
        "dial_twist_rad_s": float(T.OMEGA),
        "dial_twist_source": "the FR3 expert's 45 deg/s wrist ramp, inside the G1 rule (<= 0.9 rad/s)",
        "dial_turn_window_needed_deg": round(math.degrees(TURN_NEED), 1)}
    (rig.out_dir / "g1_plan.json").write_text(json.dumps(rig.g1_plan, indent=1))

    # ── the drive the FR3 expert calls ──────────────────────────────────────────────────────────────────────
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
        jf[arm_d] = np.clip(gravity_torques(rig.model, rig.s0.body_q.numpy(), arm_j), -budget, budget)
        rig.control.joint_f.assign(jf)

    rig.ik, rig.drive_arm = ik, drive_arm
    rig.torque_monitor = TorqueMonitor(rig.model, g1, ik.arm, {n: int(tqs[jmap[n]]) for n in ik.arm})
    G1DialExpert.finger_dofs = [int(dqs[g1["joints"][n]]) for n in fingers]
    tips = [rig.body_index(f"{PREFIX}{SIDE}_hand_Link{i}_3") for i in (1, 2)]
    ex = G1DialExpert(rig, h, DIAL, KNOB, math.radians(TARGET_DEG), g, probes, tips, a0)
    print(f"[{NAME}] seed {a.seed}: dial {np.round(axis * 1e3, 1).tolist()} mm, yaw {math.degrees(yaw0):+.1f}; "
          f"grasp yaw {math.degrees(a0):+.1f} (turn to {math.degrees(a0 + TURN_NEED):+.1f}); "
          f"ff budget {np.round(budget, 2).tolist()} N.m", flush=True)
    duration = 16.0 + abs(TARGET_DEG) / 45.0
    try:
        rig.run(min(duration, float(os.environ.get("G1_MAX_T", duration))), ex.control, ex.extra)
    except StopRun:
        pass
    paths = rig.rec.save("rollout", force_keys=["pads_N", "robot_object_contact_N"])
    return T.verdict(rig, h, g, ex, DIAL, KNOB, TARGET_DEG, key, choice, paths, a.seed, start)


if __name__ == "__main__":
    wp.init()
    main()
