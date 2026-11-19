"""A ContactBench row driven frame by frame by an external policy (clockwork-5 fork).

The scene is the task's own: the row's ``main()`` builds it exactly as ``force-bench run`` does, and ``Rig.run`` is
swapped for this module's loop (the same hook the teleop UI uses, ``teleop.scenes._capture_task``).  What the row's
scripted policy did inside its control callback is split in two:

  * the SCENERY a task applies besides the arm (the door's hidden lock probe, the bulb's holding-probe torque) is
    owned by the task adapter (``adapters.py``) and applied every frame whatever drives the arm;
  * the ARM is driven by a ``Driver``: the scripted expert (the row's own control callback, privileged: it closes its
    loops on measured mechanism state), or an external policy writing the FR3's 9 joint position targets.

The physics of one control frame is the body of ``Rig.run`` (same collision cadence, substeps, probes, recorder),
so a frame here is the same as a frame there.  An episode ends when the adapter's online success signal holds, the
driver says done, an irreversible event happens, or the world-time budget runs out; the adapter then runs its
judge phase (if any) and the world-state judge.  With ``with_upstream=True`` the row's ``main()`` continues after
the loop, so the upstream verdict (``appendix_qa.json``) is computed on the same rollout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARM_DOF = 9   # FR3: 7 arm joints (rad) + 2 finger joints (m, each finger's opening 0..0.04)


def closure(fn) -> dict:
    """The free variables a task's nested callback closes over, by name (the rig's probes, logs and handles that
    live as locals of the task's ``main()``)."""
    fn = getattr(fn, "__func__", fn)
    return {n: c.cell_contents for n, c in zip(fn.__code__.co_freevars, fn.__closure__ or ())}


@dataclass
class EpisodeEvents:
    retry_events: list = field(default_factory=list)
    irreversible: list = field(default_factory=list)
    regrasps: int = 0
    notes: list = field(default_factory=list)

    offset = staticmethod(lambda: 0.0)   # set by the Episode: the latency charged so far (robot = world + charged)

    def retry(self, t_robot, kind, **kw):
        """t_robot: the setback's robot time (the adapter stamps it when the setback happens)."""
        self.retry_events.append({"t": round(float(t_robot), 3), "kind": kind, **kw})

    def fatal(self, t, kind, **kw):
        """t: world (physics) time; recorded in robot time, the clock t_wall_s is on."""
        self.irreversible.append({"t": round(float(t) + self.offset(), 3), "kind": kind, **kw})


class _EndEpisode(Exception):
    """Raised out of the patched ``Rig.run`` to skip the row's own verdict (a policy-driven rollout of a row whose
    verdict reads the expert's internal state)."""


class Episode:
    """One episode on a built rig.  ``step(action)`` advances one control frame (1/fps s of world time)."""

    VIDEO_HZ = 15

    def __init__(self, rig, control_cb, extra_cb, adapter_cls, *, task: str, seed, cameras: bool = True,
                 video_dir=None):
        import newton
        newton.use_coord_layout_targets = True
        self.rig, self.control_cb, self.extra_cb = rig, control_cb, extra_cb
        self.task, self.seed = task, seed
        self.fps = int(rig.fps)
        self.dt = 1.0 / self.fps
        self.t = 0.0
        self.frame = 0
        self.ev = EpisodeEvents()
        self.ev.offset = lambda: self.charged_s
        self.done = False
        self.done_reason = None
        self.t_policy_end = None       # world time when the policy's part ended (before any judge phase)
        self.latency_s = 0.0           # summed policy request time (charged to the world clock in real time mode)
        self.charged_s = 0.0           # latency added to the robot clock WITHOUT stepping the world (charged mode)
        self.settle_s = 0.0            # world time stepped while a request was out, settling first (charged mode)
        self.n_requests = 0
        self.expert_driven = False
        from force_bench.policy_env.finger_force import FR3_FINGERS, FingerMagnitudes
        # a G1 row registers its own rule-A finger probes (each Dex1 jaw's two links): the episode reads those
        self.finger_mag = None if getattr(rig, "rule_a_probes", None) else \
            FingerMagnitudes(rig.model, getattr(adapter_cls, "finger_bodies", None) or FR3_FINGERS)
        self.finger_now = None
        # the FR3 arm's torques against its limits (the G1 rows carry their own TorqueMonitor)
        self.arm_torque = None
        # (the FR3's fixed base link is merged away in the simulated model; its fingers are not)
        if "fr3/fr3_leftfinger" in [str(b) for b in rig.model.body_label]:
            from force_bench.policy_env.arm_torque import ArmTorqueMonitor
            self.arm_torque = ArmTorqueMonitor(rig.model, getattr(rig, "solver", None))
        self.adapter = adapter_cls(self)
        self.cams = None
        self.videos = {}
        if cameras:
            from force_bench.policy_env.filament.cameras import policy_cameras
            self.cams = policy_cameras(rig, **self.adapter.camera_layout())    # Filament (render v4) for the G1
            if video_dir is not None:
                if hasattr(self.cams, "export_dir"):
                    self.cams.export_dir = Path(video_dir)      # the scene the policy saw, for re-rendering the rollout
                self.videos = _CameraVideos(Path(video_dir), self.cams.NAMES, self.VIDEO_HZ)
                self._record_views()

    @property
    def robot_t(self) -> float:
        """Robot time: world time stepped plus latency charged without stepping (charged mode; 0 otherwise)."""
        return self.t + self.charged_s

    def charge(self, seconds: float):
        """Charged mode (clockwork-5 D9): the robot clock moves on by ``seconds`` while the world stands still (the
        scenes are static under held targets).  The videos repeat the held frame so they stay at 1x robot time."""
        if seconds <= 0 or self.done:
            return
        self.charged_s += float(seconds)
        if self.videos:
            self.videos.hold_to(self.robot_t)
        rec = self.rig.rec
        if rec is not None and rec.frames:
            n = int(round(seconds * self.fps))
            last = rec.frames[-1]
            for _ in range(n):
                rec._video_writer().append_data(last)
            if rec.multiview and rec._writer_mv is not None:
                mv = rec._mv_host.numpy()
                for _ in range(n):
                    rec._video_writer_mv().append_data(mv)

    SETTLE_ARM_RAD_S = 0.02          # the robot has settled on its last target: every arm joint slower than this,
    SETTLE_FINGER_M_S = 0.002        # and both fingers slower than this
    SETTLE_CAP_S = 2.0               # at most this much world time is stepped before the world freezes

    def _settle_dofs(self):
        """(arm dofs, finger dofs): the FR3's [:7] / [7:9]; on a G1 rig, every G1 arm joint and every Dex1 finger
        joint, by joint label (the same thresholds)."""
        if getattr(self, "_settle_idx", None) is None:
            m = self.rig.model
            labels = [str(x or "") for x in m.joint_label]
            if any(lb.startswith("g1/") for lb in labels):
                ds = m.joint_qd_start.numpy()
                arm = [int(ds[j]) for j, lb in enumerate(labels) if lb.startswith("g1/") and any(
                    lb.endswith(f"_{k}_joint") for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                                         "wrist_roll", "wrist_pitch", "wrist_yaw"))]
                fin = [int(ds[j]) for j, lb in enumerate(labels) if lb.startswith("g1/") and "dex1_finger_joint" in lb]
                self._settle_idx = (np.array(arm, int), np.array(fin, int))
            else:
                self._settle_idx = (np.arange(7), np.arange(7, 9))
        return self._settle_idx

    def settled(self) -> bool:
        arm, fin = self._settle_dofs()
        qd = self.rig.s0.joint_qd.numpy()
        return bool(np.abs(qd[arm]).max() < self.SETTLE_ARM_RAD_S and np.abs(qd[fin]).max() < self.SETTLE_FINGER_M_S)

    def charge_settled(self, owed_s: float) -> float:
        """Charged mode (sims/clockwork/CONTRACT.md): while a request is out the world keeps stepping, last targets
        held, until the robot has settled (or SETTLE_CAP_S, or the whole owed latency, has passed); then it freezes
        and the rest of the latency is charged.  Returns the world time stepped."""
        stepped = 0.0
        limit = min(self.SETTLE_CAP_S, max(0.0, owed_s))
        while stepped + self.dt <= limit + 1e-9 and not self.done and not self.settled():
            self.step(None)
            self.adapter.check_end()
            stepped += self.dt
        self.settle_s += stepped
        self.charge(max(0.0, owed_s - stepped))
        return stepped

    # ── robot state (what a policy may see) ─────────────────────────────────────────────────────────────────────
    def joints(self) -> np.ndarray:
        return self.rig.s0.joint_q.numpy()[:ARM_DOF].astype(np.float32)

    def targets(self) -> np.ndarray:
        return self.rig.control.joint_target_q.numpy()[:ARM_DOF].astype(np.float32)

    # ── one control frame ───────────────────────────────────────────────────────────────────────────────────────
    def step(self, action=None):
        """Apply ``action`` (9 joint position targets; None holds the current targets) and advance one frame."""
        if self.done:
            raise RuntimeError("episode is over")
        self.adapter.scenery(self.t)
        if action is not None:
            self.set_targets(action)
        self._physics()

    def set_targets(self, action):
        """Write 9 joint position targets (7 arm rad, 2 finger m); they hold until the next write."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != ARM_DOF or not np.isfinite(a).all():
            raise ValueError(f"action must be {ARM_DOF} finite joint targets, got {a.shape} {a}")
        qt = self.rig.control.joint_target_q.numpy()
        qt[:7] = a[:7]
        qt[7:9] = np.clip(a[7:9], 0.0, 0.04)
        self.rig.control.joint_target_q.assign(qt)

    def step_expert(self):
        """The row's scripted expert for one frame (its own control callback, scenery included), on robot time (the
        same as world time unless latency has been charged)."""
        self.expert_driven = True
        self.control_cb(self.robot_t)    # may raise StopRun: the expert's own ending
        self._physics(scenery_done=True)

    def _physics(self, scenery_done=False):
        rig = self.rig
        sim_dt = self.dt / rig.substeps
        substep_hook = getattr(rig, "substep_hook", None)
        every = rig.effective_collide_every_substep()
        if not every:
            rig.s0.clear_forces()
            rig.pipeline.collide(rig.s0, rig.contacts)
        for _ in range(rig.substeps):
            if every:
                rig.s0.clear_forces()
                rig.pipeline.collide(rig.s0, rig.contacts)
            rig.solver.step(rig.s0, rig.s1, rig.control, rig.contacts, sim_dt)
            rig.s0, rig.s1 = rig.s1, rig.s0
            if substep_hook is not None:
                substep_hook()
        self.t += self.dt
        self.frame += 1
        rig.solver.update_contacts(rig.contacts, rig.s0)
        forces = {}
        for pname, probe in rig.probes.items():
            r = probe.sample(rig.contacts, rig.s0, self.t)
            forces[f"{pname}_N"] = round(r["force"], 4)
            forces[f"{pname}_fric_N"] = round(r["friction"], 4)
        if self.finger_mag is not None:
            self.finger_now = self.finger_mag.sample(rig.contacts, self.t)   # rule A's force (sum of magnitudes)
        else:
            self.finger_now = [rig.probes[l].history[-1]["force"] for l in rig.rule_a_probes]
        if self.arm_torque is not None:
            self.arm_torque.sample(rig.s0, rig.control, self.t)
        if self.extra_cb:
            forces.update(self.extra_cb(rig.s0, self.t))
        if rig.rec is not None:
            rig.rec.capture(rig.s0, self.t, forces=forces)
        self.adapter.monitor(self.t)
        if self.videos and self.frame % (self.fps // self.VIDEO_HZ) == 0:
            self._record_views()
        if self.frame % 60 == 0:
            print(f"[policy_env] t={self.t:6.2f}s {self.adapter.status_line()}", flush=True)

    def _record_views(self):
        """The policy cameras at 15 Hz of world time, every frame stepped: a real-time run's thinking time is in the
        video as the world it was."""
        self.videos.append(self.cams.capture(self.t)["images"])

    def close_videos(self) -> list:
        out = self.videos.close() if self.videos else []
        if self.cams is not None and hasattr(self.cams, "close"):
            self.cams.close()                    # render v4: ends the Filament server, writes its timing
        return out

    def end(self, reason: str):
        if not self.done:
            self.done, self.done_reason = True, reason
            if self.t_policy_end is None:
                self.t_policy_end = self.robot_t


class _CameraVideos:
    def __init__(self, folder: Path, names, fps):
        import imageio_ffmpeg
        self.folder, self.names, self.fps = folder, list(names), fps
        self.paths = {n: folder / f"camera_{n}.mp4" for n in self.names}
        self.w = {}
        self._ff = imageio_ffmpeg
        self.last, self.count = None, 0

    def __bool__(self):
        return True

    def hold_to(self, robot_t: float):
        """Repeat the last frame until the video covers robot_t (charged latency)."""
        while self.last is not None and self.count < int(robot_t * self.fps) + 1:
            self.append(self.last)

    def append(self, images: dict):
        self.last = images
        self.count += 1
        for n in self.names:
            img = np.ascontiguousarray(images[n])
            if n not in self.w:
                h, w = img.shape[:2]
                self.w[n] = self._ff.write_frames(str(self.paths[n]), (w, h), fps=self.fps, codec="libx264",
                                                  macro_block_size=2, output_params=["-crf", "23"])
                self.w[n].send(None)
            self.w[n].send(img)

    def close(self) -> list:
        for wr in self.w.values():
            wr.close()
        self.w = {}
        return [str(p) for p in self.paths.values() if p.exists()]


# ── running a row ───────────────────────────────────────────────────────────────────────────────────────────────────
def _row_argv(task: str, seed, extra=()):
    from force_bench.appendix_tasks.registry import BY_NAME
    from force_bench.cli import _run_plan
    if task.startswith("module:"):                 # a task module outside the registry (an embodiment variant)
        from types import SimpleNamespace
        dotted = task[len("module:"):]
        spec = SimpleNamespace(dotted=dotted, module=dotted.rsplit(".", 1)[-1])
        return spec, [*(("--seed", str(seed)) if seed is not None else ()), *extra]
    spec = BY_NAME.get(task)
    if spec is None:
        raise SystemExit(f"unknown ContactBench row {task!r} (force-bench list)")
    args = [*(("--seed", str(seed)) if seed is not None else ()), *extra]
    argv, _ = _run_plan(task, spec, args)
    return spec, argv


def run_row(task: str, seed, driver, *, budget_s: float, out_dir: Path, with_upstream: bool = True,
            cameras: bool = True, extra_argv=()) -> dict:
    """Build the row's scene, drive one episode with ``driver`` and judge it.  Returns the episode result (the
    record fields this module owns; ``record.py`` adds the contract's fields)."""
    import importlib

    from force_bench.policy_env.adapters import adapter_for
    from force_bench.rig import Rig

    spec, argv = _row_argv(task, seed, extra_argv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # the row writes under common.APPENDIX_OUT (resolved from TB_OUT_ROOT when force_bench is first imported, which
    # is before this): point it at the episode's folder for this run
    import force_bench.appendix_tasks.common as common
    appendix_out = common.APPENDIX_OUT
    common.APPENDIX_OUT = out_dir / "appendix_tasks"
    common.APPLIED_BAD_ENDING = None         # set by the row if its expert applies TB_BAD_ENDING
    adapter_cls = adapter_for(spec.module)
    mod = importlib.import_module(spec.dotted)
    box = {}
    t_host0 = time.time()

    def patched(rig_self, duration, control_cb, extra_forces_cb=None, print_every=20):
        from force_bench.appendix_tasks.common import StopRun
        ep = Episode(rig_self, control_cb, extra_forces_cb, adapter_cls, task=task, seed=seed, cameras=cameras,
                     video_dir=out_dir if cameras else None)
        box["ep"] = ep
        driver.begin(ep)
        # the expert runs to its own ending (StopRun) or the row's own duration, never cut short by the online
        # success signal: the row's verdict windows are laid out on the expert's timeline
        limit = float(duration) * getattr(driver, "duration_scale", 1.0) if driver.expert else float(budget_s)
        try:
            while not ep.done:
                if ep.robot_t >= limit - 1e-9:
                    ep.end("duration" if driver.expert else "budget")
                    break
                driver.advance(ep)       # one or more frames (a real-time request steps the world while it waits)
                if not driver.expert:
                    ep.adapter.check_end()
        except StopRun:
            ep.end("expert_finished")
        ep.adapter.judge_phase()
        box["judge"] = ep.adapter.judge()
        if common.APPLIED_BAD_ENDING is not None:   # a deliberately failed expert (solvability proof)
            box["judge"]["bad_ending"] = common.APPLIED_BAD_ENDING
        if ep.finger_mag is not None:
            series = ep.finger_mag.series
        else:                                    # a G1 row: its rule-A probes' histories (t, f0, f1)
            hs = [rig_self.probes[l].history for l in rig_self.rule_a_probes]
            series = [[hs[0][i]["t"], *(h[i]["force"] for h in hs)] for i in range(min(len(h) for h in hs))]
        np.save(out_dir / "finger_force_magnitude.npy", np.asarray(series, float))   # t, f0, f1
        box["camera_videos"] = ep.close_videos()
        if with_upstream and ep.adapter.upstream_ok():
            return rig_self.rec          # the row's main() goes on to its own verdict on this rollout
        if rig_self.rec is not None:
            box["paths"] = rig_self.rec.save("rollout")
        raise _EndEpisode

    originals = [(Rig, Rig.run)]
    Rig.run = patched
    for obj in list(vars(mod).values()):
        if isinstance(obj, type) and issubclass(obj, Rig) and obj is not Rig and "run" in obj.__dict__:
            originals.append((obj, obj.__dict__["run"]))
            obj.run = patched
    sys_argv = sys.argv
    sys.argv = [spec.dotted, *argv]
    upstream, upstream_error = None, None
    try:
        mod.main()
    except _EndEpisode:
        pass
    except Exception as exc:                 # the row's own verdict on a rollout it was not laid out for
        if "judge" not in box:
            raise
        upstream_error = f"{type(exc).__name__}: {exc}"
        print(f"[policy_env] upstream verdict failed on this rollout: {upstream_error}", flush=True)
    finally:
        sys.argv = sys_argv
        common.APPENDIX_OUT = appendix_out
        for cls, fn in originals:
            cls.run = fn
        driver.close()
    ep = box["ep"]
    qa_path = next(out_dir.rglob("appendix_qa.json"), None)
    if qa_path is not None and with_upstream and ep.adapter.upstream_ok() and upstream_error is None:
        upstream = json.loads(qa_path.read_text())
    third = sorted(str(p) for p in out_dir.rglob("rollout*.mp4"))
    return {"episode": ep, "judge": box["judge"], "upstream": upstream, "upstream_error": upstream_error,
            "videos": {"cameras": box.get("camera_videos", []), "third_person": third},
            "t_host_s": time.time() - t_host0, "row_argv": argv}
