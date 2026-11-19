"""Render version 4: the policy cameras drawn by Filament instead of Newton's raytracer.

``FilamentCameras`` is ``PolicyCameras`` with the pixels replaced: the same LEFT/RIGHT/wrist poses, the same K
(from the Newton rays) and the same ``matrix.scene-observation.v1`` calibration, so a policy sees exactly the same
geometry; only the shading changes (PBR materials, an IBL and a soft-shadowed sun, SSAO, MSAA).

The pixels come from ``filament_server`` (``server.cpp`` beside this file), one resident process per episode: it
loads the scene once and renders the three cameras per request from the current body poses.  Physics is never read
beyond ``body_q`` and never changed.

Environment (required from render version 4):
  TB_FILAMENT_SERVER   the server binary, built with build_server.sh against a Filament release
  TB_FILAMENT_IBL      a Filament IBL directory with <name>_ibl.ktx and sh.txt (e.g. the release's
                       bin/assets/ibl/lightroom_14b)
  TB_FILAMENT_LUX      "ibl,sun" intensities (optional, default "18000,75000")
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from force_bench.policy_env.cameras import GL_TO_CV, PolicyCameras

MAGIC = 0x46494C31


def _required_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"render version 4 needs {name} (see force_bench/policy_env/filament/cameras.py)")
    return os.path.expanduser(v)


class FilamentCameras(PolicyCameras):
    RENDER_VERSION = 4

    def __init__(self, rig, *, scene_fov_deg=45.0, wrist_fov_deg=80.0, **kw):
        super().__init__(rig, scene_fov_deg=scene_fov_deg, wrist_fov_deg=wrist_fov_deg, **kw)
        self.vfov = [float(scene_fov_deg), float(scene_fov_deg), float(wrist_fov_deg)]
        self._proc = None
        self._tmp = None
        self.request_ms = []

    def _apply_look(self):
        """The colours the recorder applies at its first capture (idempotent): the kit look, then render v3's."""
        from force_bench import kit_look, render_v2
        rec = getattr(self.rig, "rec", None)
        name = getattr(rec, "_look_name", None)
        if name:
            kit_look.apply_colors(self.model, name)
        render_v2.apply_colors(self.model)

    def _start(self):
        from force_bench.policy_env.filament.scene_glb import write_scene_glb
        self._apply_look()
        self._tmp = tempfile.TemporaryDirectory(prefix="filament_")
        d = Path(self._tmp.name)
        names = write_scene_glb(self.model, self.rig.s0.body_q.numpy(), str(d / "scene.glb"))
        (d / "bodies.txt").write_text("\n".join(names) + "\n")
        ibl = Path(_required_env("TB_FILAMENT_IBL"))
        ktx = next(ibl.glob("*_ibl.ktx"))
        lux = os.environ.get("TB_FILAMENT_LUX", "18000,75000").split(",")
        server = _required_env("TB_FILAMENT_SERVER")
        self._proc = subprocess.Popen([server, str(d / "scene.glb"), str(d / "bodies.txt"), str(ktx), str(ibl / "sh.txt"),
                                       f"{self.width}x{self.height}", "3", lux[0], lux[1]],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = []
        while True:
            line = self._proc.stderr.readline().decode(errors="replace")
            log.append(line)
            if "[filament] ready" in line:
                break
            if not line or self._proc.poll() is not None:
                raise RuntimeError("filament_server did not start:\n" + "".join(log[-20:]))
        self.server_log = [l.strip() for l in log if "[filament]" in l]

    def _render(self, t_s: float) -> dict:
        import time
        if self._proc is None:
            self._start()
        bq = self.rig.s0.body_q.numpy().astype(np.float32)
        worlds = [*self.scene_world, self._wrist_world(bq)]
        cams = b"".join(np.r_[M.astype(np.float32).T.reshape(-1), np.float32(f)].astype(np.float32).tobytes()
                        for M, f in zip(worlds, self.vfov))
        t0 = time.perf_counter()
        self._proc.stdin.write(struct.pack("<i", MAGIC) + bq.tobytes() + cams)
        self._proc.stdin.flush()
        n = self.width * self.height * 3
        raw = self._proc.stdout.read(3 * n)
        self.request_ms.append((time.perf_counter() - t0) * 1e3)
        if raw is None or len(raw) != 3 * n:
            raise RuntimeError("filament_server returned a short frame")
        images, calib = {}, {}
        for ci, name in enumerate(self.NAMES):
            images[name] = np.frombuffer(raw[ci * n:(ci + 1) * n], np.uint8).reshape(self.height, self.width, 3).copy()
            T = self.root_inv @ worlds[ci]
            T[:3, :3] = T[:3, :3] @ GL_TO_CV
            calib[name] = {"frame": "robot_root", "convention": "opencv", "K": self.K[ci], "T_base_camera": T.tolist(),
                           "capture_time_s": float(t_s), "resolution": [self.width, self.height],
                           "render_version": self.RENDER_VERSION}
        return {"images": images, "calibration": calib}

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(struct.pack("<i", 0))
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def policy_cameras(rig, **layout):
    """The policy cameras for the current render version: Filament from version 4, Newton's raytracer before."""
    from force_bench import render_v2
    if render_v2.version() >= FilamentCameras.RENDER_VERSION:
        return FilamentCameras(rig, **layout)
    return PolicyCameras(rig, **layout)
