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
from force_bench.g1.robot import add_rule_a_probes
from force_bench.g1.robot import (PREFIX, ContactPeak, G1IK, MagnitudeProbe, SumProbe, TorqueMonitor, add_g1, apply_look,
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


FIT_TOL = (0.002, math.radians(2.0))  # the turn's IK error allowed (the expert's xy servo trims a few mm)


def turn_fit(ik, axis, z, yaw0, gravity_joints):
    """The grasp yaw for the dial's turn: for the bar's yaw and its 180-deg twin (+ 2 pi k, kept within +-pi of 0),
    walk the top-down IK over the turn plus MARGIN at both ends (5-deg steps at the grasp height), from the top down
    and from the bottom up, each warm-started from the reset pose.  Returns (a, q at a, the largest |gravity torque|
    per arm joint over the walk) or None."""
    q_home = ik.joint_q.numpy().copy()
    p = np.array([axis[0], axis[1], z])
    for base in (yaw0, yaw0 + math.pi):
        a = math.remainder(base, 2 * math.pi)
        down = np.arange(a + TURN_NEED + MARGIN, a - MARGIN - 1e-9, -math.radians(5.0))
        for ys in (down, down[::-1]):
            ik.joint_q.assign(q_home)
            for _ in range(3):
                ik.solve(p, down_grasp_quat(ys[0]), iterations=64)
            gmax, ok, q_a = np.zeros(len(ik.arm)), True, None
            for y in ys:
                ik.solve(p, down_grasp_quat(y), iterations=32)
                if max(e / tol for e, tol in zip(ik.residual(p, down_grasp_quat(y)), FIT_TOL)) > 1.0:
                    ok = False
                    break
                gmax = np.maximum(gmax, np.abs(gravity_torques(ik.model, ik._state().body_q.numpy(), gravity_joints)))
                if abs(y - a) < 1e-6 or (q_a is None and y <= a + 1e-9 and ys is down):
                    q_a = ik.joint_q.numpy().copy()
            if ok:
                ik.joint_q.assign(q_a if q_a is not None else ik.joint_q.numpy())
                ik.solve(p, down_grasp_quat(a), iterations=48)
                return a, ik.joint_q.numpy().copy(), gmax
    return None


def reach_band(ik, axis, z, step_deg=15.0):
    """Which top-down yaws the IK reaches at the grasp (each from the reset pose, FIT_TOL): the band, as runs of
    reached yaws (deg)."""
    q_home = ik.joint_q.numpy().copy()
    p = np.array([axis[0], axis[1], z])
    ok = []
    for y in np.arange(-180.0, 180.0 + 1e-9, step_deg):
        ik.joint_q.assign(q_home)
        for _ in range(3):
            ik.solve(p, down_grasp_quat(math.radians(y)), iterations=64)
        e = ik.residual(p, down_grasp_quat(math.radians(y)))
        ok.append((float(y), e[0] <= FIT_TOL[0] and e[1] <= FIT_TOL[1]))
    runs, cur = [], None
    for y, r in ok:
        if r and cur is None:
            cur = [y, y]
        elif r:
            cur[1] = y
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    ik.joint_q.assign(q_home)
    return runs


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


REGRASP = os.environ.get("G1_DIAL_REGRASP", "0") == "1"
STROKE_MAX = math.radians(60.0)
STROKE_MIN = math.radians(10.0)
LIFT = 0.030                         # m the open hand rises between strokes (the tips clear the bar's 53 mm top)
DRIFT_TOL = math.radians(T.HALF_TICK_DEG)   # the brake must hold the dial within half a tick while the jaws are open


def band_mask(ik, axis, z, step_deg=5.0):
    """The reached top-down yaws at the grasp, sampled every step_deg, by CONTINUATION (as the arm turns): from a
    converged start every 30 deg, walk both ways in step_deg steps, warm-started, while the IK stays within FIT_TOL.
    (Independent solves from the reset pose miss yaws the walked IK reaches.)  Returns (yaws in rad, reached mask)."""
    q_home = ik.joint_q.numpy().copy()
    p = np.array([axis[0], axis[1], z])
    ys = np.radians(np.arange(-180.0, 180.0, step_deg))
    n = len(ys)
    ok = np.zeros(n, bool)

    def good(y):
        e = ik.residual(p, down_grasp_quat(float(y)))
        return e[0] <= FIT_TOL[0] and e[1] <= FIT_TOL[1]

    for s0 in range(0, n, int(round(30.0 / step_deg))):
        ik.joint_q.assign(q_home)
        for _ in range(3):
            ik.solve(p, down_grasp_quat(float(ys[s0])), iterations=64)
        if not good(ys[s0]):
            continue
        ok[s0] = True
        q_s = ik.joint_q.numpy().copy()
        for d in (1, -1):
            ik.joint_q.assign(q_s)
            for k in range(1, n):
                i = (s0 + d * k) % n
                y = ys[s0] + d * k * math.radians(step_deg)
                ik.solve(p, down_grasp_quat(float(y)), iterations=24)
                if not good(y):
                    break
                ok[i] = True
    ik.joint_q.assign(q_home)
    return ys, ok


def _runs(ys, ok):
    out, cur = [], None
    for y, r in zip(ys, ok):
        if r and cur is None:
            cur = [y, y]
        elif r:
            cur[1] = y
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def stroke_from(ys, ok, bar_abs, remaining):
    """The next stroke from the bar's absolute angle: the grasp yaw (the bar's angle or its 180-deg twin) with the
    longest reached run onward, and the stroke's turn (<= remaining, <= STROKE_MAX); None if neither side reaches
    STROKE_MIN."""
    step = ys[1] - ys[0]
    best = None
    for g in (bar_abs, bar_abs + math.pi):
        k0 = int(round((math.remainder(g, 2 * math.pi) + math.pi) / step)) % len(ys)
        run = 0
        while run < len(ys) and ok[(k0 + run) % len(ys)]:
            run += 1
        reach = (run - 1) * step if run else -1.0
        if best is None or reach > best[1]:
            best = (math.remainder(g, 2 * math.pi), reach)
    if best[1] < STROKE_MIN:
        return None
    return best[0], min(remaining, STROKE_MAX, best[1] - math.radians(5.0))


class G1DialRegraspExpert(G1DialExpert):
    """The FR3 expert in strokes (the dial's brake holds it between them): each stroke closes on the bar at its
    current angle (the FR3 close: squeeze, xy servo), turns it closed on the measured angle (the FR3 turn) to the
    stroke's end; between strokes the jaws open, the hand rises LIFT, turns to the next grasp yaw at <= WRIST_MAX and
    lowers; the last stroke ends with the FR3 trim, open, retreat and still.  Per stroke: the turn and the dial's
    drift while the jaws were open."""

    def setup_strokes(self, ys, ok, bar_yaw0, first):
        self.ys, self.ok, self.bar_yaw0 = ys, ok, bar_yaw0
        self.strokes, self.final = [], False
        self.target_final = self.target
        self._begin(first[0], first[1], 0.0, None)

    def _begin(self, grasp_abs, turn, ang, drift):
        """A stroke from dial angle ang: the hand's yaw command is relative to (grasp_abs - ang)."""
        self.yaw0 = grasp_abs - ang
        self.st["yaw_cmd"] = ang
        self.stroke_end = ang + turn
        self.final = self.stroke_end >= self.target_final - 1e-6
        self.target = self.target_final if self.final else self.stroke_end
        self.strokes.append({"grasp_yaw_deg": round(math.degrees(grasp_abs), 1), "from_deg": round(math.degrees(ang), 2),
                             "planned_turn_deg": round(math.degrees(turn), 1), "drift_while_open_deg":
                             None if drift is None else round(math.degrees(drift), 2)})
        self.rig.dial_strokes = self.strokes

    def control(self, t):
        st = self.st
        ph = st["phase"]
        ang, _ = T.D.knob_coords(self.rig, self.h)
        if ph == "trim" and not self.final:            # a stroke's end (not the last): let go and regrasp
            self.strokes[-1]["turned_deg"] = round(math.degrees(ang) - self.strokes[-1]["from_deg"], 2)
            st["r_ang"], st["r_yaw"], st["r_z"] = ang, self.yaw0 + st["yaw_cmd"], self.tcp0[2]
            self.go("r_open", t)
            ph = "r_open"
        tau = t - st["t0"]
        z0 = self.tcp0[2]
        if ph == "r_open":
            u = T.ease(tau / T.T_OPEN)
            self._drive_abs(st["r_yaw"], z0, self.f_grip + (self.f_open - self.f_grip) * u)
            if tau >= T.T_OPEN:
                self.go("r_lift", t)
            return
        if ph == "r_lift":
            u = T.ease(tau / 0.6)
            self._drive_abs(st["r_yaw"], z0 + LIFT * u, self.f_open)
            if tau >= 0.6:
                remaining = self.target_final - ang
                nxt = stroke_from(self.ys, self.ok, self.bar_yaw0 + ang, remaining)
                if nxt is None:                        # no side reaches onward: stop here (the judge scores it)
                    self.final, self.yaw0 = True, 0.0
                    self.strokes.append({"stopped": "no grasp side reaches onward", "at_deg": round(math.degrees(ang), 2)})
                    st["yaw_release"], st["z_release"] = st["r_yaw"], z0 + LIFT
                    self.go("retreat", t)
                    return
                st["r_next"] = nxt
                d = math.remainder(nxt[0] - st["r_yaw"], 2 * math.pi)
                st["r_from"], st["r_to"] = st["r_yaw"], st["r_yaw"] + d
                st["r_T"] = max(0.6, 1.875 * abs(d) / WRIST_MAX)
                self.go("r_rotate", t)
            return
        if ph == "r_rotate":
            u = T.ease(tau / st["r_T"])
            self._drive_abs(st["r_from"] + (st["r_to"] - st["r_from"]) * u, z0 + LIFT, self.f_open)
            if tau >= st["r_T"]:
                self.go("r_lower", t)
            return
        if ph == "r_lower":
            u = T.ease(tau / 0.6)
            self._drive_abs(st["r_to"], z0 + LIFT * (1 - u), self.f_open)
            if tau >= 0.6:
                drift = ang - st["r_ang"]
                self._begin(st["r_to"], st["r_next"][1], ang, drift)
                self.go("close", t)
            return
        if ph in ("open", "retreat", "still") and self.strokes and "turned_deg" not in self.strokes[-1]:
            self.strokes[-1]["turned_deg"] = round(math.degrees(ang) - self.strokes[-1]["from_deg"], 2)
        super().control(t)

    def _drive_abs(self, yaw_abs, z, f):
        p = self.tcp0[:2] + self.xy
        self.rig.drive_arm(np.array([p[0], p[1], z]), down_grasp_quat(yaw_abs), fingers=f)


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
    if not REGRASP:                                          # the FR3 row's own finger drive (the single-grasp
        for d in fdofs:                                      # set); the regrasp set keeps the Dex1's own gain:
            rig.builder.joint_target_ke[d] = T.FINGER_KE     # 20 kN/m on the Dex1 fingers chattered through the bar
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
    if REGRASP:                                              # strokes: the band, and the first stroke from it
        ys_b, ok_b = band_mask(ik, axis, tcp0[2])
        first = stroke_from(ys_b, ok_b, yaw0, math.radians(TARGET_DEG))
        fit = None
        if first is not None:
            q_home = ik.joint_q.numpy().copy()
            for _ in range(3):
                ik.solve(tcp0, down_grasp_quat(first[0]), iterations=64)
            gm = np.abs(gravity_torques(ik.model, ik._state().body_q.numpy(), ik.arm_joints)) + 0.5
            fit = (first[0], ik.joint_q.numpy().copy(), gm)
            ik.joint_q.assign(q_home)
    else:
        fit = turn_fit(ik, axis, tcp0[2], yaw0, ik.arm_joints)
    miss = None
    if fit is None:
        if REGRASP:
            LAST_MISS.update(dial_xy_m=[round(float(v), 4) for v in axis], dial_yaw_deg=round(math.degrees(yaw0), 1),
                             arm=SIDE, binding_constraint="wrist-band position vs the bar's yaw: neither grasp side of "
                             "the bar's start angle reaches a 10-deg stroke onward",
                             reached_band_deg=[[round(math.degrees(v), 0) for v in r] for r in _runs(ys_b, ok_b)])
        need = [[round(math.degrees(math.remainder(b, 2 * math.pi) - MARGIN), 1),
                 round(math.degrees(math.remainder(b, 2 * math.pi) + TURN_NEED + MARGIN), 1)]
                for b in (yaw0, yaw0 + math.pi)]
        LAST_MISS.update({} if REGRASP else dict(needed_deg=round(math.degrees(TURN_NEED + 2 * MARGIN), 1)))
        if not REGRASP:
            LAST_MISS.update(dial_xy_m=[round(float(v), 4) for v in axis], dial_yaw_deg=round(math.degrees(yaw0), 1),
                         needed_deg=round(math.degrees(TURN_NEED + 2 * MARGIN), 1), arm=SIDE,
                         ik_tolerance=[FIT_TOL[0], round(math.degrees(FIT_TOL[1]), 1)],
                         binding_constraint="wrist-band position vs the bar's yaw: neither needed yaw range lies "
                                            "inside the reached top-down band",
                         needed_yaw_ranges_deg=need, reached_band_deg=reach_band(ik, axis, tcp0[2]))
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
    # one finger = a Dex1 jaw's two links; its force is the SUM OF CONTACT MAGNITUDES on them against the knob (a net
    # vector total let a finger box chattering through the 10 mm bar read 549 N while its contacts summed 8.3 kN)
    probes = {"finL": MagnitudeProbe(rig.model, links[1], counterpart_shapes=knob_shapes),
              "finR": MagnitudeProbe(rig.model, links[2], counterpart_shapes=knob_shapes),
              "robot_object": RobotContactProbe(rig.model, moving, static_shapes=static_labels)}
    rig.probes.update(probes)
    rig.pad_probes = ("finL", "finR")
    add_rule_a_probes(rig, SIDE)                            # rule A: the sum of contact magnitudes
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
        "dial_finger_stiffness_N_per_m": T.FINGER_KE if not REGRASP else float(rig.model.joint_target_ke.numpy()[fdofs[0]]),
        "dial_finger_stiffness_source": ("the FR3 dial row's own finger position-drive gain (a controller gain, not a "
                                         "material property): the row closes 1.5 x squeeze / gain past contact")
                                        if not REGRASP else
                                        ("the Dex1's own finger gain (matrix's MuJoCo Dex1, as the other G1 rows; a "
                                         "controller gain): the FR3 row's 20 kN/m chattered the Dex1 finger boxes "
                                         "through the bar; the grip closes 1.25 x squeeze / gain past contact so the "
                                         "70 N cap binds"),
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
    if REGRASP:
        ex = G1DialRegraspExpert(rig, h, DIAL, KNOB, math.radians(TARGET_DEG), g, probes, tips, a0)
        kp_f = float(rig.model.joint_target_ke.numpy()[fdofs[0]])
        ex.f_grip = max(0.0, ex.f_touch - 1.25 * T.SQUEEZE[KNOB] / kp_f)   # the cap (70 N) binds at the Dex1's gain
        rig.finger_params = finger_readback(rig.solver, rig.model, g1, SIDE, overdrive_m=1.25 * T.SQUEEZE[KNOB] / kp_f)
        ex.setup_strokes(ys_b, ok_b, yaw0, first)
        rig.variant_extra.update(
            dial_regrasp=True, dial_stroke_max_deg=math.degrees(STROKE_MAX), dial_regrasp_lift_m=LIFT,
            dial_drift_tolerance_deg=math.degrees(DRIFT_TOL),
            dial_regrasp_note="strokes of <= 60 deg, each grasped at the bar's current angle or its 180-deg twin "
                              "(whichever the measured top-down band continues from); the jaws open, the hand rises "
                              "30 mm and turns to the next grasp at <= 0.9 rad/s between strokes; the brake holds "
                              "the dial (drift checked against half a tick)",
            dial_reached_band_deg=[[round(math.degrees(v), 0) for v in r] for r in
                                   _runs(ys_b, ok_b)])
    else:
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
