"""Per-row adapters: scenery, online events and success signal, judge phase and world-state judge.

Each judge keeps the row's own thresholds and replaces only what read the scripted expert's internal state or its
timeline (see JUDGE.md for every criterion, upstream vs here).  Detection rules for the episode record:

  regrasp     both pads above GRIP_N for GRIP_FRAMES frames after having been below it (the literal count)
  setback     the task's own progress went backwards (clockwork-5 D1): the latch re-extended before the door had
              opened, the door closed back below 5 deg after opening, the bulb backed out > 30 deg below its best,
              or the bulb was re-gripped after it had been measured tight and released (a re-tighten)
  retry       a setback followed by a redo (progress rising again or a re-grip); stroke re-grips with progress
              holding are not retries (clockwork-5 D1)
  irreversible
              forced_door     the leaf opened >= 1 deg before the latch had ever cleared the strike (the row's own
                              "unlocked_by_knob_then_opened" can no longer pass)
              part_dropped    the bulb is off the socket axis by > 30 mm, or 30 mm below the socket top
              force_limit     a pad (door) or the summed robot-bulb contact (bulb) above FORCE_LIMIT_N for
                              FORCE_LIMIT_FRAMES frames: 4x the finger drive's own effort ceiling
              sim_diverged    a non-finite state
"""

from __future__ import annotations

import numpy as np

from force_bench.policy_env.episode import closure

GRIP_N, GRIP_FRAMES = 2.0, 3        # the door row's pads_check gate (2 N in every 3-frame window)
BACKOUT_DEG = 30.0
FORCE_LIMIT_N, FORCE_LIMIT_FRAMES = 150.0, 4


class _Run:
    """Debounced boolean: true after ``n`` consecutive true samples, false after ``n`` false ones."""

    def __init__(self, n):
        self.n, self.v, self.k = n, False, 0

    def update(self, x: bool) -> int:
        """Returns +1 on a rise, -1 on a fall, 0 otherwise."""
        if x != self.v:
            self.k += 1
            if self.k >= self.n:
                self.v, self.k = x, 0
                return 1 if x else -1
        else:
            self.k = 0
        return 0


class Adapter:
    name = "base"

    def __init__(self, ep):
        self.ep = ep
        self.rig = ep.rig
        self.setback = None
        self.success_online = False
        self.over = 0
        self.grip = _Run(GRIP_FRAMES)
        self.n_grips = 0

    # what the env applies besides the arm
    def scenery(self, t):
        pass

    def camera_layout(self) -> dict:
        rec = self.rig.rec
        tf = rec.cam_tf.numpy()[0][0]
        return {"scene_pos": np.array(tf[:3], float), "scene_target": np.asarray(rec.cam2_target, float)}

    def monitor(self, t):
        q = self.rig.s0.joint_q.numpy()
        if not np.isfinite(q).all():
            self.ep.ev.fatal(t, "sim_diverged")
            self.ep.end("irreversible")

    def _grip_update(self, t, gripped: bool):
        ch = self.grip.update(gripped)
        if ch > 0:
            self.n_grips += 1
            if self.n_grips > 1:
                self.ep.ev.regrasps += 1
                self._redo(t, "regrasp")
        return ch

    def _set_back(self, t, kind, **kw):
        if self.setback is None:
            self.setback = {"t": t, "kind": kind, **kw}

    def _redo(self, t, how):
        if self.setback is not None:
            self.ep.ev.retry(self.setback["t"], self.setback["kind"], redo=how, t_redo=round(float(t), 3))
            self.setback = None

    def _force(self, t, f):
        self.over = self.over + 1 if f > FORCE_LIMIT_N else 0
        if self.over == FORCE_LIMIT_FRAMES:
            self.ep.ev.fatal(t, "force_limit", peak_N=round(float(f), 1), limit_N=FORCE_LIMIT_N)
            self.ep.end("irreversible")

    def check_end(self):
        pass

    def judge_phase(self):
        pass

    def judge(self) -> dict:
        raise NotImplementedError

    def upstream_ok(self) -> bool:
        return True

    def status_line(self) -> str:
        return ""

    def progress(self) -> float | None:
        return None


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 5.1  turn_doorknob_open_door_flat
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
class DoorAdapter(Adapter):
    name = "door"
    HOLD_S = 1.0           # the row's motion_stopped window: the last 1 s, TCP drift < 3 mm, door range < 1 deg

    def __init__(self, ep):
        super().__init__(ep)
        from force_bench.kit.mechanisms.doorknob_module import JAMB_CLEAR_MM
        c, x = closure(ep.control_cb), closure(ep.extra_cb)
        self.probe = c["probe"]                     # HingeProbe: the hidden lock probe's torque at the hinge
        self.log, self.tcp_log = x["log"], x["tcp_log"]   # (t, door_th, knob_ph, latch_x), (t, tcp xyz)
        self.jamb_mm = JAMB_CLEAR_MM
        self.t_clear = None
        self.max_knob = 0.0
        self.max_door = 0.0
        self.max_latch = 0.0
        self._reext = self._closed = False
        self.door_deg = self.knob_deg = self.latch_mm = 0.0
        self.pads = (0.0, 0.0)

    def scenery(self, t):
        if not self.ep.expert_driven:
            from force_bench.appendix_tasks._door_kit import LOCK_PROBE_NM, LOCK_PROBE_S
            self.probe.set(-LOCK_PROBE_NM if LOCK_PROBE_S[0] <= t < LOCK_PROBE_S[1] else 0.0)

    def monitor(self, t):
        super().monitor(t)
        if self.ep.done or not self.log:
            return
        _, th, ph, x = self.log[-1]
        self.door_deg, self.knob_deg, self.latch_mm = float(np.degrees(-th)), float(np.degrees(-ph)), x * 1e3
        pl = self.rig.probes["fr3/fr3_leftfinger"].history[-1]["force"]
        pr = self.rig.probes["fr3/fr3_rightfinger"].history[-1]["force"]
        self.pads = (pl, pr)
        self.max_latch = max(self.max_latch, self.latch_mm)
        self._force(t, max(pl, pr))
        self._grip_update(t, min(pl, pr) > GRIP_N)
        if self.t_clear is None and self.latch_mm > self.jamb_mm:
            self.t_clear = t
        if self.t_clear is None and self.door_deg >= 1.0:
            self.ep.ev.fatal(t, "forced_door", door_deg=round(self.door_deg, 2),
                             note="the leaf opened >= 1 deg before the latch cleared the strike")
            self.ep.end("irreversible")
            return
        # progress: the knob angle until the latch clears, then the door angle
        if self.knob_deg > self.max_knob + 10.0 or self.door_deg > self.max_door + 2.0:
            self._redo(t, "progress")
        self.max_knob = max(self.max_knob, self.knob_deg)
        self.max_door = max(self.max_door, self.door_deg)
        # D1 setbacks: the latch re-extends (the knob let go or turned back before the door opened), or the door
        # closes back below 5 deg after it had been opened
        if self.t_clear is not None and self.latch_mm < self.jamb_mm and self.max_door < 5.0 and not self._reext:
            self._reext = True
            self._set_back(t, "latch_reextended", knob_deg=round(self.knob_deg, 1))
        if self.latch_mm >= self.jamb_mm:
            self._reext = False
        if self.max_door >= 5.0 and self.door_deg < 5.0 and not self._closed:
            self._closed = True
            self._set_back(t, "door_closed_back", door_deg=round(self.door_deg, 1), max_door_deg=round(self.max_door, 1))
            self.max_door = self.door_deg
        if self.door_deg >= 5.0:
            self._closed = False
        self.success_online = self._held_open(self.HOLD_S)

    def _held_open(self, window) -> bool:
        if self.t_clear is None or not self.grip.v or self.ep.t < window:
            return False
        t = np.array([r[0] for r in self.log[-int(window * self.ep.fps) - 1:]])
        th = np.degrees(-np.array([r[1] for r in self.log[-len(t):]]))
        tcp = np.array([r[1:] for r in self.tcp_log[-len(t):]])
        drift = float(np.linalg.norm(tcp[-1] - tcp[0]) * 1000)
        return bool(t[-1] - t[0] >= window - 1.5 / self.ep.fps and th.min() >= 5.0 and np.ptp(th) < 1.0
                    and drift < 3.0)

    def check_end(self):
        if self.success_online:
            self.ep.end("success")

    def progress(self):
        # partial credit for diagnosis only: half for unlocking, half for the door's way to 5 deg
        unl = 1.0 if self.t_clear is not None else min(1.0, max(0.0, self.max_latch) / self.jamb_mm)
        return round(0.5 * unl + 0.5 * min(1.0, max(0.0, self.door_deg) / 5.0), 3)

    def status_line(self):
        return (f"door={self.door_deg:5.1f}deg knob={self.knob_deg:5.1f}deg latch={self.latch_mm:5.2f}mm "
                f"pads={self.pads[0]:.1f}/{self.pads[1]:.1f}N")

    def judge(self) -> dict:
        """The row's four criteria on world state.  Same thresholds; the windows are anchored to the episode's own
        events (latch clear, episode end) instead of the expert's schedule T."""
        from force_bench.appendix_tasks._door_kit import lock_probe_check, pads_check
        arr = np.array(self.log)
        t_a, th_a, ph_a, x_a = arr.T
        open_deg = np.degrees(-th_a)
        t_end = float(t_a[-1])
        crit = {"locked_at_start": lock_probe_check(t_a, th_a)}
        eng = x_a * 1000.0 > self.jamb_mm
        t_unl = float(t_a[np.argmax(eng)]) if eng.any() else float("inf")
        pre = open_deg[t_a < t_unl]
        crit["unlocked_by_knob_then_opened"] = {
            "pass": bool(eng.any() and (pre.max() if len(pre) else 0.0) < 1.0 and open_deg[-1] >= 5.0),
            "t_latch_clear_s": round(t_unl, 3), "door_deg_before_latch_clear_max": round(float(pre.max()) if len(pre)
                                                                                         else 0.0, 3),
            "final_door_deg": round(float(open_deg[-1]), 2), "peak_door_deg": round(float(open_deg.max()), 2),
            "threshold_deg": 5.0, "upstream": "identical"}
        if eng.any():
            crit["finger_grip_sustained"] = {**pads_check(self.rig, t_unl, t_end),
                                             "upstream": "window from the expert's squeeze + 0.2 s (T[2]+0.2) to "
                                                         "T[-1]; here from latch clear to the episode end: the "
                                                         "door is lifted through the knob"}
        else:
            crit["finger_grip_sustained"] = {"pass": False, "note": "the latch never cleared"}
        tcp = np.array(self.tcp_log)
        m = tcp[:, 0] >= t_end - 1.0
        drift = float(np.linalg.norm(tcp[m, 1:][-1] - tcp[m, 1:][0]) * 1000)
        dd = float(np.ptp(open_deg[t_a >= t_end - 1.0]))
        crit["motion_stopped_door_held_open"] = {
            "pass": bool(drift < 3.0 and dd < 1.0 and open_deg[-1] >= 5.0),
            "tcp_drift_last_1s_mm": round(drift, 2), "door_range_last_1s_deg": round(dd, 3),
            "upstream": "identical (the last 1 s of the episode)"}
        finite = bool(np.isfinite(arr).all())
        return {"pass": bool(all(v["pass"] for v in crit.values()) and finite), "criteria": crit,
                "finite": finite, "t_end_s": round(t_end, 3)}


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 5.2  install_light_bulb (in hand)
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
class BulbAdapter(Adapter):
    name = "bulb"
    PAD_N = 1.0               # the row's real_finger_grasp gate (mean pad force > 1 N while twisting)
    THREAD_TOUCH_N = 0.05
    CLEAR_S = 0.4             # hand clear this long (and moved away) = the policy has let go for good
    AWAY_M = 0.025
    TWIST_RATE = np.radians(2.0)

    def __init__(self, ep):
        super().__init__(ep)
        import force_bench.appendix_tasks._kit_thread as kt
        from force_bench.appendix_tasks.common import CONTACT_FREE_N
        self.kt = kt
        self.ex = ep.extra_cb.__self__              # the row's ScrewExpert: its log is the world-state log
        self.S, self.P = self.ex.S, self.ex.P
        self.cols = {c: i for i, c in enumerate(kt.LOG_COLS)}
        self.free_n = CONTACT_FREE_N
        self.probe_tau = 0.0
        self.t_judge = None
        self.y_touch = None
        self.t_touch = None
        self.t_seat = None
        self.max_turns = 0.0
        self.hold = None
        self.t_tight = None
        self.y_tight = None
        self.released_tight = False
        self.clear_since = None
        self.d0 = None
        self.gripping = _Run(GRIP_FRAMES)

    def _c(self, row, name):
        return row[self.cols[name]]

    def scenery(self, t):
        if self.ep.expert_driven:
            return
        ex = self.ex
        by = ex.part_yaw.update(ex.row())         # the expert's control callback keeps this up to date
        if ex._by_prev is not None:
            ex.omega_f = 0.8 * ex.omega_f + 0.2 * (by - ex._by_prev) * self.ep.fps
        ex._by_prev = by
        ex.st["probe"] = self.probe_tau
        self.rig.solver.probe = self.probe_tau

    def turns(self, yaw):
        return 0.0 if self.y_touch is None else float(-(yaw - self.y_touch) / (2 * np.pi))

    def monitor(self, t):
        super().monitor(t)
        if self.ep.done or not self.ex.log or self.t_judge is not None:
            return
        kt, r = self.kt, self.ex.log[-1]
        c = lambda n: float(self._c(r, n))  # noqa: E731
        fL, fR, yaw = c("fL"), c("fR"), c("yaw")
        self._force(t, c("robot"))
        ch = self.gripping.update(min(fL, fR) > self.PAD_N)
        if ch > 0:
            self.n_grips += 1
            if self.n_grips > 1:
                self.ep.ev.regrasps += 1
                if self.released_tight:
                    self._set_back(t, "retighten", note="re-gripped after it had been measured tight and released")
                    self.released_tight, self.t_tight = False, None
                self._redo(t, "regrasp")
        if ch < 0 and self.t_tight is not None:
            self.released_tight = True
        # off the socket: dropped
        if c("axis_err") > 0.030 or c("Z") < -0.030:
            self.ep.ev.fatal(t, "part_dropped", axis_err_mm=round(c("axis_err") * 1e3, 1), Z_mm=round(c("Z") * 1e3, 1))
            self.ep.end("irreversible")
            return
        if self.y_touch is None and c("thread") > self.THREAD_TOUCH_N:
            self.y_touch, self.t_touch = yaw, t
        if self.t_seat is None and self.t_touch is not None and c("seat") > kt.F_SEAT_TOUCH:
            self.t_seat = t
        n = self.turns(yaw)
        if n > self.max_turns + 10.0 / 360:
            self._redo(t, "progress")
        if n < self.max_turns - BACKOUT_DEG / 360:
            self._set_back(t, "back_out", turns=round(n, 3), max_turns=round(self.max_turns, 3))
            self.max_turns = n
        self.max_turns = max(self.max_turns, n)
        # tight, measured as the row measures it: seated, pad torque >= TAU_TIGHT with the flanks preloaded, then
        # the part stalled (< STALL_HOLD_DEG) over T_TIGHT_CONFIRM
        pre = -c("flank_z") >= kt.PRELOAD_TIGHT
        if self.t_tight is None and self.t_seat is not None:
            if self.hold is None and c("tau_pads") >= kt.TAU_TIGHT and pre:
                self.hold = (t, yaw)
            elif self.hold is not None and t - self.hold[0] >= kt.T_TIGHT_CONFIRM:
                if abs(yaw - self.hold[1]) < np.radians(kt.STALL_HOLD_DEG) and pre:
                    self.t_tight, self.y_tight = t, yaw
                self.hold = None
        # the hand has left for good: clear of the bulb for CLEAR_S and moved AWAY_M from where it held it
        if not self.ep.expert_driven:
            d = float(np.linalg.norm(self.ex.tcp_meas() - self.ex.row()[:3]))
            self.d0 = d if self.d0 is None else self.d0
            if c("robot") < self.free_n and d > self.d0 + self.AWAY_M:
                self.clear_since = t if self.clear_since is None else self.clear_since
            else:
                self.clear_since = None
        self.success_online = self.t_tight is not None and self.clear_since is not None

    def check_end(self):
        if self.clear_since is not None and self.ep.t - self.clear_since >= self.CLEAR_S:
            self.ep.end("released" if self.t_tight is not None else "released_untight")

    def judge_phase(self):
        """After the policy's part, the env checks the bulb as the row does once its hand has left: still for
        T_STILL, then the holding probe PROBE_TAU loosen-way and tighten-way for T_PROBE each.  The arm holds its
        last targets.  Skipped for an irreversible ending (scored 0 regardless) and for the expert (it runs its own
        identical sequence)."""
        ep, kt = self.ep, self.kt
        if ep.expert_driven or ep.ev.irreversible:
            return
        ep.done = False
        self.t_judge = ep.t
        for dur, tau in ((kt.T_STILL, 0.0), (kt.T_PROBE, +kt.PROBE_TAU), (kt.T_PROBE, -kt.PROBE_TAU), (0.3, 0.0)):
            self.probe_tau = tau
            for _ in range(int(round(dur * ep.fps))):
                ep.step(None)
        self.probe_tau = 0.0
        ep.done = True

    def upstream_ok(self) -> bool:
        return self.ep.expert_driven        # the row's verdict reads the expert's state (tight_reason, strokes)

    def progress(self):
        return round(min(1.0, max(0.0, self.max_turns) / max(1e-6, self.P.revs)), 3)

    def status_line(self):
        r = self.ex.log[-1] if self.ex.log else None
        if r is None:
            return ""
        return (f"turns={self.turns(self._c(r, 'yaw')):+.3f} seat={self._c(r, 'seat'):.2f}N "
                f"tau={self._c(r, 'tau_pads') * 1e3:.0f}mNm pads={self._c(r, 'fL'):.1f}/{self._c(r, 'fR'):.1f}N "
                f"tight={'y' if self.t_tight else 'n'}")

    def judge(self) -> dict:
        kt, S, P = self.kt, self.S, self.P
        L = np.array(self.ex.log)
        C = {c: L[:, i] for i, c in enumerate(kt.LOG_COLS)}
        t = C["t"]
        crit = {}
        g0 = t < 0.5
        z0 = S.p0[2] - S.fixture_world[2, 3]
        crit["held_in_hand_at_start"] = {
            "pass": bool(min(C["fL"][g0].mean(), C["fR"][g0].mean()) > 2.0 and abs(C["Z"][g0].min() - z0) < 0.001),
            "mean_pad_N_first_0.5s": [round(float(C["fL"][g0].mean()), 2), round(float(C["fR"][g0].mean()), 2)],
            "upstream": "identical"}
        # threaded in: lead of the measured helix and the turns, from the first thread contact
        touch = np.flatnonzero(C["thread"] > self.THREAD_TOUCH_N)
        i_t = int(touch[0]) if len(touch) else None
        seat = np.flatnonzero((C["seat"] > kt.F_SEAT_TOUCH) & (np.arange(len(t)) >= (i_t or 0))) if i_t is not None \
            else np.array([], int)
        i_s = int(seat[0]) if len(seat) else None
        tight_i = self._tight_index(C, i_s)
        lead = turns = None
        if i_t is not None:
            yt = C["yaw"][i_t]
            w = np.arange(len(t)) >= i_t
            w &= np.arange(len(t)) <= (i_s if i_s is not None else len(t) - 1)
            w &= -(C["yaw"] - yt) >= 0.25 * 2 * np.pi        # past the first quarter turn: the thread has caught
            if w.sum() > 20:
                lead = abs(np.polyfit(C["yaw"][w], C["Z"][w], 1)[0]) * 2 * np.pi
            end_i = tight_i if tight_i is not None else len(t) - 1
            turns = float(-(C["yaw"][end_i] - yt) / (2 * np.pi))
        min_turns = kt.MIN_TURNS_FRAC * P.revs
        crit["bulb_threaded_in"] = {
            "pass": bool(lead is not None and abs(lead - P.lead) <= kt.LEAD_TOL * P.lead and turns is not None
                         and turns >= min_turns),
            "measured_lead_mm_per_turn": None if lead is None else round(lead * 1e3, 3),
            "cad_lead_mm_per_turn": round(P.lead * 1e3, 3), "turns_first_contact_to_tight":
                None if turns is None else round(turns, 3), "min_turns": round(min_turns, 2),
            "upstream": "same thresholds; fit window from a quarter turn past the first thread contact (upstream: "
                        "the expert's own engagement detection) to the seat contact"}
        # the judge phase, located on the probe column (the expert's and the env's sequence are the same)
        pr = np.flatnonzero(C["probe"] > 0)
        tp = float(t[pr[0]]) if len(pr) else None
        pre = (t >= (tp or 1e9) - 0.4) & (t < (tp or 1e9))
        gap = float(C["seat_gap"][pre].mean()) if pre.any() else 1.0
        preload = float(-C["flank_z"][pre].mean()) if pre.any() else 0.0
        tilt = float(C["tilt"][pre].mean()) if pre.any() else 99.0
        rate = float(np.abs(np.diff(C["yaw"][pre])).sum() / max(1e-6, 0.4)) if pre.sum() > 1 else None
        crit["seated_and_tightened"] = {
            "pass": bool(tight_i is not None and gap <= kt.SEAT_GAP_MAX and preload >= kt.PRELOAD_MIN
                         and tilt <= kt.TILT_MAX_DEG and rate is not None and rate < kt.STALL_RATE),
            "tight_measured_at_s": None if tight_i is None else round(float(t[tight_i]), 3),
            "seat_gap_after_release_mm": round(gap * 1e3, 3), "preload_after_release_N": round(preload, 2),
            "tilt_deg": round(tilt, 3), "part_rate_after_release_deg_s": None if rate is None else
            round(float(np.degrees(rate)), 2),
            "upstream": "upstream requires the expert's own stop reason ('seat torque'); here the same tightness "
                        "condition MEASURED on the rollout: seated, pad torque >= TAU_TIGHT with flank preload >= "
                        "PRELOAD_TIGHT, then the part turning < STALL_HOLD_DEG over T_TIGHT_CONFIRM; the "
                        "after-release thresholds are identical"}
        moves = {}
        for k, nm in ((0, "loosen_way"), (1, "tighten_way")):
            w = (t >= (tp or 1e9) + k * kt.T_PROBE) & (t < (tp or 1e9) + (k + 1) * kt.T_PROBE)
            moves[nm] = {"rotation_deg": round(float(np.degrees(np.ptp(C["yaw"][w]))), 4) if w.any() else None}
        crit["stationary_under_small_torque"] = {
            "pass": bool(tp is not None and all(v["rotation_deg"] is not None and v["rotation_deg"] < kt.STATIONARY_DEG
                                                for v in moves.values())), **moves, "upstream": "identical"}
        rate_y = np.gradient(C["yaw"], t) if len(t) > 2 else np.zeros_like(t)
        tw = (rate_y < -self.TWIST_RATE) & (C["robot"] > self.free_n) & (t < (tp or 1e9))
        crit["real_finger_grasp"] = {
            "pass": bool(tw.any() and C["fL"][tw].mean() > 1.0 and C["fR"][tw].mean() > 1.0),
            "mean_pad_N_while_twisting": [round(float(C["fL"][tw].mean()), 2), round(float(C["fR"][tw].mean()), 2)]
            if tw.any() else None,
            "upstream": "upstream averages over the expert's engage/twist strokes; here over every frame the hand "
                        "is on the bulb while it turns tightening-way faster than 2 deg/s"}
        t_from = (tp - kt.T_STILL) if tp is not None else float(t[-1])
        m = t >= t_from
        peak = float(C["robot"][m].max()) if m.any() else float("inf")
        crit["released_and_clear"] = {"pass": bool(m.any() and peak < self.free_n),
                                      "robot_object_contact_peak_N": round(peak, 4),
                                      "window_s": [round(t_from, 2), round(float(t[-1]), 2)],
                                      "upstream": "identical threshold; window from the judge phase's start"}
        finite = bool(np.isfinite(L).all())
        return {"pass": bool(all(v["pass"] for v in crit.values()) and finite), "criteria": crit, "finite": finite,
                "t_end_s": round(float(t[-1]), 3)}

    def _tight_index(self, C, i_s):
        """First frame at which the measured tightness condition completes (see monitor())."""
        kt = self.kt
        if i_s is None:
            return None
        t, yaw = C["t"], C["yaw"]
        ok = (C["tau_pads"] >= kt.TAU_TIGHT) & (-C["flank_z"] >= kt.PRELOAD_TIGHT)
        n = int(round(kt.T_TIGHT_CONFIRM * self.ep.fps))
        for i in np.flatnonzero(ok):
            if i < i_s or i + n >= len(t):
                continue
            j = i + n
            if abs(yaw[j] - yaw[i]) < np.radians(kt.STALL_HOLD_DEG) and -C["flank_z"][j] >= kt.PRELOAD_TIGHT:
                return int(j)
        return None


ADAPTERS = {"turn_doorknob_open_door_flat": DoorAdapter, "install_light_bulb": BulbAdapter}


def adapter_for(module: str):
    if module not in ADAPTERS:
        raise SystemExit(f"no policy adapter for {module!r} yet (have: {sorted(ADAPTERS)})")
    return ADAPTERS[module]
