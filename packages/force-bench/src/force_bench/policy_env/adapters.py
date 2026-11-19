"""Per-row adapters: scenery, online events and success signal, judge phase and world-state judge.

Each judge keeps the row's own thresholds and replaces only what read the scripted expert's internal state or its
timeline (see JUDGE.md for every criterion, upstream vs here).  Detection rules for the episode record:

  regrasp     both pads above GRIP_N for GRIP_FRAMES frames after having been below it (the literal count)
  setback     the task's own progress went backwards (clockwork-5 D1): the latch re-extended before the door had
              opened, the door closed back below 5 deg after opening, the bulb backed out > 30 deg below its best,
              or the bulb was re-gripped after it had been measured tight and released (a re-tighten)
  retry       a setback followed by a redo (progress rising again or a re-grip); stroke re-grips with progress
              holding are not retries (clockwork-5 D1)
  irreversible (ends a policy's episode; recorded but not stopping the expert, whose upstream verdict needs its whole
  rollout)
              forced_door     the leaf opened >= 1 deg before the latch had ever cleared the strike (the row's own
                              "unlocked_by_knob_then_opened" can no longer pass)
              part_dropped    the bulb is off the socket axis by > 30 mm, or 30 mm below the socket top
              force_limit     either finger pad above a FIXED FORCE_LIMIT_N (150 N) for FORCE_LIMIT_FRAMES frames,
                              whatever the grip setting (the user's rule)
              sim_diverged    a non-finite state
"""

from __future__ import annotations

import os

import numpy as np

from force_bench.policy_env.episode import closure

GRIP_N, GRIP_FRAMES = 2.0, 3        # the door row's pads_check gate (2 N in every 3-frame window)
BACKOUT_DEG = 30.0
ENVELOPE_SQUEEZE_LIMIT_N = 25.0     # bulb glass, report-only and provisional (unmeasured)
FORCE_LIMIT_N = 150.0
# How long a finger must stay above FORCE_LIMIT_N for force_limit to fire. 4 consecutive frames (67 ms at 60 Hz) is
# the rule every FR3 run used; 1 is "any single frame". An explicit setting (TB_FORCE_LIMIT_FRAMES) until the contract
# fixes the rule; every record carries force_limit_rule.
FORCE_LIMIT_FRAMES = int(os.environ.get("TB_FORCE_LIMIT_FRAMES", "4"))


def force_limit_rule() -> str:
    return (f"per-finger contact force (normal-magnitude-sum: the sum of each contact's normal-force magnitude on the finger body's shapes, contract #1773, 19 Nov) > {FORCE_LIMIT_N:g} N for "
            f"{FORCE_LIMIT_FRAMES} consecutive frame(s) at 60 Hz, fixed, independent of the grip setting")


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


def _elapsed(col, log, i, k) -> float:
    """Logged time from row i to row k (the expert's own clock, accumulated floats)."""
    return float(col(log[k], "t")) - float(col(log[i], "t"))


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
            self._fatal(t, "sim_diverged")

    def _grip_update(self, t, gripped: bool):
        ch = self.grip.update(gripped)
        if ch > 0:
            self.n_grips += 1
            if self.n_grips > 1:
                self.ep.ev.regrasps += 1
                self._redo(t, "regrasp")
        return ch

    def _set_back(self, t, kind, **kw):
        if self.setback is None:                       # stamped in robot time (world + latency charged so far)
            self.setback = {"t": t + self.ep.charged_s, "kind": kind, **kw}

    def _redo(self, t, how):
        if self.setback is not None:
            self.ep.ev.retry(self.setback["t"], self.setback["kind"], redo=how,
                             t_redo=round(float(t + self.ep.charged_s), 3))
            self.setback = None

    def force_limit(self, summed: bool = False) -> float:
        """The irreversible contact-force limit: a FIXED 150 N on either finger pad (the user's rule, in the contract).
        Independent of the grip setting: an earlier hand-relative limit (4x the finger effort ceiling) rose silently
        with the squeeze (a G1 at 80 N got 320 N per pad)."""
        return FORCE_LIMIT_N

    def _force(self, t, f, summed=False):
        """f: the larger of the two NET pad forces this frame (the row's probes), kept as peak_pad_net_n.  Rule A reads
        the episode's sum-of-contact-magnitudes finger force (contract #1773), which net cancellation can't hide."""
        lim = FORCE_LIMIT_N
        self.peak_pad_net_n = max(getattr(self, "peak_pad_net_n", 0.0), float(f))
        if self.ep.finger_now is not None:
            f = max(self.ep.finger_now)
        self.peak_pad_n = max(getattr(self, "peak_pad_n", 0.0), float(f))
        self.over = self.over + 1 if f > lim else 0
        if self.over == FORCE_LIMIT_FRAMES:
            self._fatal(t, "force_limit", peak_N=round(float(f), 1), limit_N=lim, measured="per pad")

    def _fatal(self, t, kind, **kw):
        """An irreversible event ends a policy's episode; the expert's is recorded but runs to its own end (its
        upstream verdict needs the whole rollout)."""
        self.ep.ev.fatal(t, kind, **kw)
        if not self.ep.expert_driven:
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
        self.t_below1 = 0.0          # the last frame the door was shut (< 1 deg): the final opening starts there
        self._pad_hist = []          # the last GRIP_FRAMES frames of (left, right) pad force
        self.grip_run_start = None   # since when both pads pass the row's grip gate (pads_check's 3-frame rule)

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
        if self.door_deg < 1.0:
            self.t_below1 = t
        # pads_check's gate, online: each pad's rolling 3-frame max above GRIP_N, as a continuous run
        self._pad_hist = (self._pad_hist + [(pl, pr)])[-GRIP_FRAMES:]
        if len(self._pad_hist) == GRIP_FRAMES and min(max(h[0] for h in self._pad_hist),
                                                     max(h[1] for h in self._pad_hist)) > GRIP_N:
            if self.grip_run_start is None:
                self.grip_run_start = t - (GRIP_FRAMES - 1) / self.ep.fps
        else:
            self.grip_run_start = None
        self._force(t, max(pl, pr))
        self._grip_update(t, min(pl, pr) > GRIP_N)
        if self.t_clear is None and self.latch_mm > self.jamb_mm:
            self.t_clear = t
        if self.t_clear is None and self.door_deg >= 1.0:
            if not any(e["kind"] == "forced_door" for e in self.ep.ev.irreversible):
                self._fatal(t, "forced_door", door_deg=round(self.door_deg, 2),
                            note="the leaf opened >= 1 deg before the latch cleared the strike")
            if self.ep.done:
                return
        # progress: the knob angle until the latch clears, then the door angle
        if self.knob_deg > self.max_knob + 10.0 or self.door_deg > self.max_door + 2.0:
            self._redo(t, "progress")
        self.max_knob = max(self.max_knob, self.knob_deg)
        self.max_door = max(self.max_door, self.door_deg)
        # setbacks (the contract's retry rule): the latch re-extends (the knob let go or turned back before the door opened), or the door
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

    def _anchor(self, t_clear) -> float:
        """Where the grip window starts: the latch clearing, or the start of the final opening (the last frame the
        door was shut) if that is later.  A retried opening is not failed for an earlier loss of grip (that costs
        attempts); the opening that counts is held through the knob."""
        return max(t_clear, self.t_below1)

    def _held_open(self, window) -> bool:
        """The online success signal = the judge would pass now: unlocked, both pads gripping without a break since
        the final opening began, and the door >= 5 deg, still, with the TCP still, for the last 1 s."""
        if self.t_clear is None or self.grip_run_start is None or self.ep.t < window:
            return False
        if self.grip_run_start > self._anchor(self.t_clear) + 1.5 / self.ep.fps:
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
            shut = np.flatnonzero(open_deg < 1.0)
            t_shut = float(t_a[shut[-1]]) if len(shut) else 0.0
            t0 = max(t_unl, t_shut)
            crit["finger_grip_sustained"] = {**pads_check(self.rig, t0, t_end),
                                             "upstream": "window from the expert's squeeze + 0.2 s (T[2]+0.2) to "
                                                         "T[-1]; here from the latch clearing, or the start of the "
                                                         "final opening (the last frame the door was shut) if later, "
                                                         "to the episode end: the opening that counts is lifted "
                                                         "through the knob, and a retried opening costs attempts"}
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
        self.hold, self.hold_j = None, None
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
        self._force(t, max(fL, fR))
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
            if not any(e["kind"] == "part_dropped" for e in self.ep.ev.irreversible):
                self._fatal(t, "part_dropped", axis_err_mm=round(c("axis_err") * 1e3, 1), Z_mm=round(c("Z") * 1e3, 1))
            if self.ep.done:
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
        # (sampled as the judge's _tight_index: hold from frame i, yaw one frame later, preload at the confirm frame)
        k = len(self.ex.log) - 1
        if self.t_tight is None and self.t_seat is not None:
            if self.hold is None and c("tau_pads") >= kt.TAU_TIGHT and -c("flank_z") >= kt.PRELOAD_TIGHT:
                self.hold, self.hold_j = k, None
            elif self.hold is not None:
                i = self.hold
                if self.hold_j is None and _elapsed(self._c, self.ex.log, i, k) >= kt.T_TIGHT_CONFIRM:
                    self.hold_j = k                    # the confirm frame, timed as the expert times it
                if self.hold_j is not None and k >= self.hold_j + 1:
                    j = self.hold_j
                    y0 = float(self._c(self.ex.log[i + 1], "yaw"))
                    y1 = float(self._c(self.ex.log[j + 1], "yaw"))
                    if abs(y1 - y0) < np.radians(kt.STALL_HOLD_DEG) and \
                            -float(self._c(self.ex.log[j], "flank_z")) >= kt.PRELOAD_TIGHT:
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
        # report-only, provisional: how hard the fingers squeeze the glass.  The kit bulb is one body gripped by its
        # globe (the FR3 and G1 experts both pinch the envelope), so the per-finger force on the bulb is the force on
        # the envelope.  25 N is unmeasured (2.5x the observed peak); it flags, it never fails or scores.
        env = float(np.maximum(C["fL"], C["fR"]).max()) if len(t) else 0.0
        qa = {"max_envelope_finger_force_N": round(env, 2), "envelope_squeeze_limit_N": ENVELOPE_SQUEEZE_LIMIT_N,
              "envelope_squeeze_limit_note": "provisional: unmeasured; 2.5x the observed peak",
              "flags": ["max_envelope_finger_force_N"] if env > ENVELOPE_SQUEEZE_LIMIT_N else []}
        return {"pass": bool(all(v["pass"] for v in crit.values()) and finite), "criteria": crit, "finite": finite,
                "t_end_s": round(float(t[-1]), 3), "physics_qa": qa}

    def _tight_index(self, C, i_s):
        """First frame at which the row's tightness condition completes, sampled as the row's expert samples it: the
        torque and preload of frame i start the hold; the part's yaw is read one frame later (the log's yaw column
        is the state the next control step sees), over T_TIGHT_CONFIRM; the preload is checked again at the confirm
        frame.  (Reading both at frame i counted a spike's own settling as turning: seed 4 measured 1.01 deg where
        the expert, and the physics after the spike, had 0.41 deg.)"""
        kt = self.kt
        if i_s is None:
            return None
        yaw = np.r_[C["yaw"][1:], C["yaw"][-1:]]          # the yaw of the state after frame k
        ok = (C["tau_pads"] >= kt.TAU_TIGHT) & (-C["flank_z"] >= kt.PRELOAD_TIGHT)
        t = C["t"]
        for i in np.flatnonzero(ok):
            if i < i_s:
                continue
            # the confirm frame as the expert finds it: the first whose logged time is T_TIGHT_CONFIRM past the
            # hold's start, on the same accumulated float times (9 frames of 1/60 s can sum to 0.149999..., so
            # round(0.15 * fps) frames confirms one frame early: FR3 5.2 seed 1's rerun read a preload dip there)
            later = np.flatnonzero(t[i:] - t[i] >= kt.T_TIGHT_CONFIRM)
            if not len(later):
                continue
            j = i + int(later[0])
            if j >= len(t) - 1:
                continue
            if abs(yaw[j] - yaw[i]) < np.radians(kt.STALL_HOLD_DEG) and -C["flank_z"][j] >= kt.PRELOAD_TIGHT:
                return int(j)
        return None



# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 5.6  turn_dial_to_target (continuous "standard" dial; push and snap later, as variants)
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
class DialAdapter(Adapter):
    """The row's Expert object is used only as the world-state logger (its ``extra``: knob angle, press, pads, robot
    contact, fingertip midpoint); its phases and timestamps are never read by the judge."""
    name = "dial"
    TURN_RATE = np.radians(2.0)     # turning = the knob moves faster than this, with the hand on it
    CLEAR_S = 0.4

    def __init__(self, ep):
        super().__init__(ep)
        import force_bench.appendix_tasks.turn_dial_to_target as dt
        from force_bench.appendix_tasks.common import CONTACT_FREE_N
        self.dt = dt
        self.ex = ep.extra_cb.__self__
        self.target = float(self.ex.target)                 # rad, relative to the dial's 0 mark
        self.dial = self.ex.dial
        self.tol = np.radians(dt.SNAP_TOL_DEG if self.dial == "snap" else dt.HALF_TICK_DEG)
        self.free_n = CONTACT_FREE_N
        self.best_err = None
        self.contact_run = _Run(GRIP_FRAMES)
        self.release_tips = None
        self.clear_since = None
        self.t_judge = None

    def monitor(self, t):
        super().monitor(t)
        if self.ep.done or not self.ex.log or self.t_judge is not None:
            return
        r = self.ex.log[-1]
        self._force(t, max(r["fL"], r["fR"]))
        ch = self._grip_update(t, min(r["fL"], r["fR"]) > GRIP_N)
        err = abs(r["ang"] - self.target)
        if self.best_err is None or err < self.best_err - np.radians(10.0):
            if self.best_err is not None:
                self._redo(t, "progress")
            self.best_err = err if self.best_err is None else min(self.best_err, err)
        self.best_err = min(self.best_err, err)
        if err > self.best_err + np.radians(BACKOUT_DEG):
            self._set_back(t, "dial_turned_away", error_deg=round(float(np.degrees(err)), 1),
                           best_error_deg=round(float(np.degrees(self.best_err)), 1))
            self.best_err = err
        # the hand has let go for good: no contact for CLEAR_S and MOVE_MIN away from where the contact ended
        tips = np.asarray(r["tips"], float)
        touching = r["robot"] >= self.free_n
        if touching:
            self.release_tips, self.clear_since = tips, None
        elif not self.ep.expert_driven:
            away = self.release_tips is None or np.linalg.norm(tips - self.release_tips) >= self.dt.MOVE_MIN
            if away:
                self.clear_since = t if self.clear_since is None else self.clear_since
            else:
                self.clear_since = None
        self.success_online = err <= self.tol and self.clear_since is not None

    def check_end(self):
        if self.clear_since is not None and self.ep.t - self.clear_since >= self.CLEAR_S:
            self.ep.end("released")

    def judge_phase(self):
        """The row's own ending after the hand has gone: still for T_STILL, arm holding its last targets."""
        ep = self.ep
        if ep.expert_driven or ep.ev.irreversible:
            return
        ep.done = False
        self.t_judge = ep.t
        for _ in range(int(round(self.dt.T_STILL * ep.fps))):
            ep.step(None)
        ep.done = True

    def upstream_ok(self) -> bool:
        return self.ep.expert_driven        # the row's verdict reads the expert's phases (turn/trim, open, still)

    def progress(self):
        if self.best_err is None:
            return 0.0
        return round(max(0.0, 1.0 - self.best_err / max(1e-6, abs(self.target))), 3)

    def final_error_deg(self):
        return float(np.degrees(self.ex.log[-1]["ang"] - self.target)) if self.ex.log else None

    def status_line(self):
        r = self.ex.log[-1] if self.ex.log else None
        if r is None:
            return ""
        return (f"dial={np.degrees(r['ang']):+6.1f}deg target={np.degrees(self.target):+.0f} "
                f"pads={r['fL']:.1f}/{r['fR']:.1f}N robot={r['robot']:.2f}N")

    def judge(self) -> dict:
        dt = self.dt
        L = self.ex.log
        t = np.array([r["t"] for r in L])
        ang = np.array([r["ang"] for r in L])
        robot = np.array([r["robot"] for r in L])
        pads = np.array([r["fL"] + r["fR"] for r in L])
        tips = np.array([r["tips"] for r in L])
        crit = {}
        # the still window: the judge phase (policy), or the expert's own still phase (its last T_STILL s)
        t_still = self.t_judge if self.t_judge is not None else float(t[-1]) - dt.T_STILL
        still = t >= t_still - 1e-9
        err = float(np.degrees(ang[-1] - self.target))
        crit["dial_at_target"] = {"pass": bool(abs(err) <= np.degrees(self.tol)),
                                  "target_deg": round(float(np.degrees(self.target)), 2),
                                  "final_deg": round(float(np.degrees(ang[-1])), 2), "error_deg": round(err, 2),
                                  "tolerance_deg": round(float(np.degrees(self.tol)), 2), "upstream": "identical"}
        if self.dial == "snap":
            crit["dial_at_target"]["note"] = "snap dial: detent seating (pad deflection) is not judged here yet"
        drift = float(np.degrees(np.ptp(ang[still]))) if still.any() else float("nan")
        crit["dial_stays_after_release"] = {"pass": bool(np.isfinite(drift) and drift < dt.DRIFT_MAX_DEG),
                                            "drift_still_deg": round(drift, 3), "upstream": "identical threshold; "
                                            "window = the still phase after the hand has gone"}
        rate = np.abs(np.gradient(ang, t)) if len(t) > 2 else np.zeros_like(t)
        turning = (rate > self.TURN_RATE) & (robot >= self.free_n) & ~still
        crit["finger_grasp_real"] = {"pass": bool(turning.any() and pads[turning].mean() > 2.0),
                                     "mean_pad_force_N": round(float(pads[turning].mean()), 2) if turning.any() else 0.0,
                                     "upstream": "upstream averages over the expert's turn/trim phases; here over "
                                                 "every frame the hand is on the knob while it turns > 2 deg/s"}
        touch = np.flatnonzero(robot >= self.free_n)
        i_rel = int(touch[-1]) if len(touch) else 0
        moved = float(np.linalg.norm(tips[-1] - tips[i_rel]))
        peak = float(robot[still].max()) if still.any() else float("inf")
        crit["released_and_clear"] = {"pass": bool(still.any() and peak < self.free_n and moved >= dt.MOVE_MIN),
                                      "robot_object_contact_peak_N": round(peak, 4), "hand_moved_mm": round(moved * 1e3, 1),
                                      "min_move_mm": dt.MOVE_MIN * 1e3, "upstream": "identical thresholds; the move is "
                                      "measured from the last contact (upstream: from the expert's open phase)"}
        finite = bool(np.isfinite(ang).all())
        h = getattr(self.ex, "h", None)            # the dial kit's handle: which collider the robot touched on the bar
        bar = (getattr(h, "info", {}) or {}).get("bar_collider") if h is not None else None
        return {"pass": bool(all(v["pass"] for v in crit.values()) and finite), "criteria": crit, "finite": finite,
                "t_end_s": round(float(t[-1]), 3), "final_error_deg": round(err, 2),
                "dial_bar_collider": bar or {"type": "mesh", "source": "the knob's CAD mesh (dial_knob_flat.stl)"}}


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 5.7  screw_nut_onto_screw (nut in the pinch, 0.6 turn caught on the bolt's first thread)
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
class NutAdapter(Adapter):
    """The row's log (t, nut z, robot contact, left pad, right pad) is the world-state log; the expert's plan, stroke
    windows and t_plan are never read by the judge."""
    name = "nut"
    CLEAR_S = 0.4
    AWAY_M = 0.025
    DESCEND = 0.05e-3          # m/s: the nut is screwing down (helix descent) faster than this

    def __init__(self, ep):
        super().__init__(ep)
        import force_bench.appendix_tasks.screw_nut_onto_screw as sn
        from force_bench.appendix_tasks.common import CONTACT_FREE_N
        from force_bench.tasks.nut_bolt import LEAD
        self.sn, self.lead = sn, float(LEAD)
        x = closure(ep.extra_cb)
        self.log, self.ni = x["log"], int(x["ni"])
        self.free_n = CONTACT_FREE_N
        self.fl, self.fr = self.rig.body_index("fr3/fr3_leftfinger"), self.rig.body_index("fr3/fr3_rightfinger")
        self.min_z = float(sn.Z_START)
        self.yaw = None              # the nut's unwrapped rotation about the bolt axis, rad
        self._yaw_prev = None
        self.best_turn = 0.0         # best tighten-way rotation, turns (progress for setbacks: z bobs in the thread play)
        self.clear_since = None
        self.d0 = None
        self.t_judge = None

    def _tips(self):
        bq = self.rig.s0.body_q.numpy()
        return 0.5 * (bq[self.fl, :3] + bq[self.fr, :3]), bq[self.ni, :3]

    def turns(self, z=None):
        return (float(self.sn.Z_START) - (self.min_z if z is None else z)) / self.lead

    def engagement_mm(self, z):
        """How much of the nut's height is below the bolt's thread top (7.02 mm = fully engaged)."""
        sn = self.sn
        return float(np.clip(sn.THREAD_TOP - (z - sn.NUT_H / 2), 0.0, sn.NUT_H) * 1e3)

    def monitor(self, t):
        super().monitor(t)
        if self.ep.done or not self.log or self.t_judge is not None:
            return
        _, z, rc, fl, fr = self.log[-1]
        self._force(t, max(fl, fr))
        self._grip_update(t, min(fl, fr) > 1.0)
        tips, nut = self._tips()
        if np.hypot(nut[0], nut[1]) > 0.010 or z > float(self.sn.Z_START) + 0.008 or z < -0.02:
            if not any(e["kind"] == "part_dropped" for e in self.ep.ev.irreversible):
                self._fatal(t, "part_dropped", nut_xy_mm=[round(float(v) * 1e3, 1) for v in nut[:2]],
                            nut_z_mm=round(float(z) * 1e3, 2))
            if self.ep.done:
                return
        # progress on the nut's ROTATION (tighten-way = clockwise from above, negative yaw): its height bobs in the
        # thread play by more than 30 deg of lead whenever the pads let go, so height would see false back-outs
        q = self.rig.s0.body_q.numpy()[self.ni][3:7]
        a = 2.0 * np.arctan2(q[2], q[3])
        if self._yaw_prev is None:
            self.yaw, self._yaw0 = a, a
        else:
            self.yaw += (a - self._yaw_prev + np.pi) % (2 * np.pi) - np.pi
        self._yaw_prev = a
        turn = float(-(self.yaw - self._yaw0) / (2 * np.pi))
        if turn > self.best_turn + 10.0 / 360:
            self._redo(t, "progress")
        if turn < self.best_turn - BACKOUT_DEG / 360:
            self._set_back(t, "back_out", turns=round(turn, 3), best_turns=round(self.best_turn, 3))
            self.best_turn = turn
        self.best_turn = max(self.best_turn, turn)
        self.min_z = min(self.min_z, z)
        if not self.ep.expert_driven:
            d = float(np.linalg.norm(tips - nut))
            self.d0 = d if self.d0 is None else self.d0
            if rc < self.free_n and d > self.d0 + self.AWAY_M:
                self.clear_since = t if self.clear_since is None else self.clear_since
            else:
                self.clear_since = None
        self.success_online = z <= self.sn.ENGAGED_Z and self.clear_since is not None

    def check_end(self):
        if self.clear_since is not None and self.ep.t - self.clear_since >= self.CLEAR_S:
            self.ep.end("released" if self.log[-1][1] <= self.sn.ENGAGED_Z else "released_untight")

    def judge_phase(self):
        """The row's ending after the hand has gone: still for T_STILL, arm holding its last targets."""
        ep = self.ep
        if ep.expert_driven or ep.ev.irreversible:
            return
        ep.done = False
        self.t_judge = ep.t
        for _ in range(int(round(self.sn.T_STILL * ep.fps))):
            ep.step(None)
        ep.done = True

    def upstream_ok(self) -> bool:
        return self.ep.expert_driven          # the row's verdict is laid out on the expert's plan (t_plan, strokes)

    def progress(self):
        span = float(self.sn.Z_START - self.sn.ENGAGED_Z)
        return round(float(np.clip((self.sn.Z_START - self.min_z) / span, 0.0, 1.0)), 3)

    def status_line(self):
        if not self.log:
            return ""
        _, z, rc, fl, fr = self.log[-1]
        return (f"nut_z={z * 1e3:6.2f}mm turns={self.turns(z):+.2f} engaged={self.engagement_mm(z):.2f}mm "
                f"pads={fl:.1f}/{fr:.1f}N robot={rc:.2f}N")

    def judge(self) -> dict:
        sn = self.sn
        L = np.array(self.log)
        t, z, rc, fl, fr = (L[:, k] for k in range(5))
        crit = {}
        g0 = t < sn.T_GRIP0 + 0.3
        crit["held_in_hand_at_start"] = {
            "pass": bool(min(fl[g0].max(), fr[g0].max()) > 1.0 and abs(z[g0].min() - sn.Z_START) < 0.001),
            "pad_peak_N": [round(float(fl[g0].max()), 2), round(float(fr[g0].max()), 2)], "upstream": "identical"}
        turns = (sn.Z_START - float(z.min())) / self.lead
        crit["threads_fully_engaged"] = {
            "pass": bool(float(z[-1]) <= sn.ENGAGED_Z), "final_nut_centre_mm": round(float(z[-1]) * 1e3, 2),
            "engaged_when_centre_below_mm": round(sn.ENGAGED_Z * 1e3, 2), "turns_screwed": round(turns, 2),
            "engagement_depth_mm": round(self.engagement_mm(float(z[-1])), 2),
            "nut_height_mm": round(sn.NUT_H * 1e3, 2), "upstream": "identical"}
        grip = np.flatnonzero((fl > 1.0) & (fr > 1.0))
        i_rel = int(grip[-1]) if len(grip) else 0
        t_rel = float(t[i_rel])
        after = (t >= t_rel + 0.1) & (t <= t_rel + 0.4)       # just after the pads have let go
        z_rel = float(np.median(z[after])) if after.any() else float(z[i_rel])
        crit["nut_stays_put_after_release"] = {
            "pass": bool(abs(float(z[-1]) - z_rel) < 0.0003), "z_before_release_mm": round(z_rel * 1e3, 3),
            "final_z_mm": round(float(z[-1]) * 1e3, 3), "release_at_s": round(t_rel, 3),
            "upstream": "identical threshold: the nut's height just after the pads let go (0.1-0.4 s after the last "
                        "frame both gripped > 1 N) against its final height. Upstream compares the end of the expert's "
                        "plan, also after the release; measuring before the release would count the last stroke's own "
                        "descent as movement"}
        dz = np.gradient(z, t) if len(t) > 2 else np.zeros_like(t)
        screwing = (dz < -self.DESCEND) & (rc >= self.free_n)
        crit["real_finger_grasp"] = {
            "pass": bool(screwing.any() and fl[screwing].mean() > 1.0 and fr[screwing].mean() > 1.0),
            "mean_pad_N_while_screwing": [round(float(fl[screwing].mean()), 2), round(float(fr[screwing].mean()), 2)]
            if screwing.any() else None,
            "upstream": "upstream averages over the expert's strokes; here over every frame the nut screws down "
                        "with the hand on it"}
        t_still = self.t_judge if self.t_judge is not None else float(t[-1]) - sn.T_STILL
        still = t >= t_still - 1e-9
        peak = float(rc[still].max()) if still.any() else float("inf")
        crit["released_and_clear"] = {"pass": bool(still.any() and peak < self.free_n),
                                      "robot_object_contact_peak_N": round(peak, 4),
                                      "window_s": [round(t_still, 2), round(float(t[-1]), 2)],
                                      "upstream": "identical threshold; window = the still phase"}
        finite = bool(np.isfinite(L).all())
        return {"pass": bool(all(v["pass"] for v in crit.values()) and finite), "criteria": crit, "finite": finite,
                "t_end_s": round(float(t[-1]), 3), "turns": round(turns, 2),
                "engagement_depth_mm": round(self.engagement_mm(float(z[-1])), 2)}


ADAPTERS = {"turn_doorknob_open_door_flat": DoorAdapter, "turn_door_handle_open_door_flat": DoorAdapter,
            "install_light_bulb": BulbAdapter, "turn_dial_to_target": DialAdapter,
            "screw_nut_onto_screw": NutAdapter}


def adapter_for(module: str):
    if module not in ADAPTERS:
        raise SystemExit(f"no policy adapter for {module!r} yet (have: {sorted(ADAPTERS)})")
    return ADAPTERS[module]
