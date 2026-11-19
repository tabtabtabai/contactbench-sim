"""Render version 4: the G1 policy cameras drawn by Filament in the Isaac challenge scene.

``FilamentCameras`` is ``PolicyCameras`` with the pixels replaced: the same cameras (the scene pair and the robot's
mounts, e.g. the G1 head d435 and wrist), the same K (from the Newton rays) and the same ``matrix.scene-observation.v1``
calibration, so a policy sees exactly the same geometry.  Only the image changes: the Isaac room, table, lights and
G1 materials (``isaac_scene``), PBR shading, a soft-shadowed key light, SSAO, screen-space reflections, MSAA.

The pixels come from ``filament_server`` (``server.cpp`` here, built by ``build_server.sh`` against a pinned Filament
release), one resident process per episode: it loads the scene once as a glb and renders every camera per request
from the current body poses.  Physics is never read beyond ``body_q`` and never changed.

Selection (``policy_cameras``): G1 rigs only.  ``TB_RENDER_VERSION=4`` asks for it (and fails if the server is
missing); with ``TB_RENDER_VERSION`` unset it is the G1 default whenever ``TB_FILAMENT_SERVER`` names the binary;
otherwise Newton's raytracer (render version 1).

Environment: TB_FILAMENT_SERVER (the binary), TB_RENDER_VERSION (optional, see above).
Each episode's scene is saved beside its camera videos (render_v4_scene.glb, render_v4_bodies.txt, render_v4.json), so
any recorded rollout can be re-rendered from its rollout_states.npz with exactly the scene the policy saw.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from force_bench.policy_env.cameras import GL_TO_CV, PolicyCameras
from force_bench.policy_env.filament import isaac_scene

MAGIC = 0x46494C31
RENDER_VERSION = 4


def _is_g1(rig) -> bool:
    return any(str(b).startswith("g1/") for b in rig.model.body_label)


def render_version(rig=None) -> int:
    """The render version policy cameras use for this rig (4 = Filament in the Isaac scene, 1 = Newton)."""
    v = os.environ.get("TB_RENDER_VERSION")
    if rig is not None and not _is_g1(rig):
        return 1
    if v:
        return int(v)
    return RENDER_VERSION if os.environ.get("TB_FILAMENT_SERVER") else 1


def server_options(key_travel) -> list[str]:
    L = isaac_scene.LOOK
    kd = ",".join(f"{v:.4f}" for v in key_travel)
    return [str(L["ibl"]), str(L["sun_lux"]), "ibl_color=" + ",".join(map(str, isaac_scene.DOME["color"])),
            f"sun_dir={kd}", "sun_color=" + ",".join(map(str, isaac_scene.KEY["color"])),
            f"sun_radius_deg={isaac_scene.KEY['angle_deg'] / 2}", "skybox=1", f"ev100={L['ev100']}",
            f"tonemap={L['tonemap']}", f"ssr={L['ssr']}"]


class FilamentCameras(PolicyCameras):
    RENDER_VERSION = RENDER_VERSION

    def __init__(self, rig, *, scene_fov_deg=45.0, wrist_fov_deg=80.0, **kw):
        super().__init__(rig, scene_fov_deg=scene_fov_deg, wrist_fov_deg=wrist_fov_deg, **kw)
        self.vfov = [float(scene_fov_deg), float(scene_fov_deg), *[float(m[3]) for m in self.mounts]]
        self._proc = None
        self._tmp = None
        self.export_dir = None            # the episode sets this to its video folder: the scene is saved there
        self.request_ms = []

    def _apply_look(self):
        """The kit colours the recorder applies at its first capture (idempotent)."""
        from force_bench import kit_look
        name = getattr(getattr(self.rig, "rec", None), "_look_name", None)
        if name:
            kit_look.apply_colors(self.model, name)

    def _start(self):
        from force_bench.policy_env.filament.scene_glb import write_scene_glb
        server = os.environ.get("TB_FILAMENT_SERVER")
        if not server or not Path(os.path.expanduser(server)).exists():
            raise RuntimeError("render version 4 needs TB_FILAMENT_SERVER: the filament_server binary "
                               "(force_bench/policy_env/filament/build_server.sh)")
        self._apply_look()
        self._tmp = tempfile.TemporaryDirectory(prefix="filament_")
        d = Path(self._tmp.name)
        info = write_scene_glb(self.model, self.rig.s0.body_q.numpy(), str(d / "scene.glb"))
        (d / "bodies.txt").write_text("\n".join(info["names"]) + "\n")
        opts = server_options(info["key_travel"])
        self._proc = subprocess.Popen([os.path.expanduser(server), str(d / "scene.glb"), str(d / "bodies.txt"), "-", "-",
                                       f"{self.width}x{self.height}", str(len(self.NAMES)), *opts],
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
        self.meta = {"render_version": RENDER_VERSION, "renderer": "Filament (filament_server)", "server_options": opts,
                     "cameras": {n: {"vfov_deg": f, "K": k} for n, f, k in zip(self.NAMES, self.vfov, self.K)},
                     "resolution": [self.width, self.height], "look": isaac_scene.LOOK,
                     "scene": "unibot-scene-spec.md Isaac room + table placed by g1/pelvis; Isaac G1 materials"}
        if self.export_dir is not None:
            ex = Path(self.export_dir)
            ex.mkdir(parents=True, exist_ok=True)
            shutil.copy(d / "scene.glb", ex / "render_v4_scene.glb")
            shutil.copy(d / "bodies.txt", ex / "render_v4_bodies.txt")
            (ex / "render_v4.json").write_text(json.dumps(self.meta, indent=1))

    def _render(self, t_s: float) -> dict:
        if self._proc is None:
            self._start()
        bq = self.rig.s0.body_q.numpy().astype(np.float32)
        worlds = [*self.scene_world, *self._robot_worlds(bq)]
        cams = b"".join(np.r_[M.astype(np.float32).T.reshape(-1), np.float32(f)].astype(np.float32).tobytes()
                        for M, f in zip(worlds, self.vfov))
        t0 = time.perf_counter()
        self._proc.stdin.write(struct.pack("<i", MAGIC) + bq.tobytes() + cams)
        self._proc.stdin.flush()
        n = self.width * self.height * 3
        raw = self._proc.stdout.read(len(self.NAMES) * n)
        self.request_ms.append((time.perf_counter() - t0) * 1e3)
        if raw is None or len(raw) != len(self.NAMES) * n:
            raise RuntimeError("filament_server returned a short frame")
        images, calib = {}, {}
        for ci, name in enumerate(self.NAMES):
            images[name] = np.frombuffer(raw[ci * n:(ci + 1) * n], np.uint8).reshape(self.height, self.width, 3).copy()
            T = self.root_inv @ worlds[ci]
            T[:3, :3] = T[:3, :3] @ GL_TO_CV
            calib[name] = {"frame": "robot_root", "convention": "opencv", "K": self.K[ci], "T_base_camera": T.tolist(),
                           "capture_time_s": float(t_s), "resolution": [self.width, self.height],
                           "render_version": RENDER_VERSION}
        return {"images": images, "calibration": calib}

    def timing(self) -> dict:
        t = np.asarray(self.request_ms[3:] or self.request_ms, float)
        return {"cameras": len(self.NAMES), "resolution": [self.width, self.height], "requests": len(self.request_ms),
                "ms_median": round(float(np.median(t)), 2) if len(t) else None,
                "ms_p90": round(float(np.percentile(t, 90)), 2) if len(t) else None}

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(struct.pack("<i", 0))
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        if self.export_dir is not None and self.request_ms:
            p = Path(self.export_dir) / "render_v4.json"
            if p.exists():
                meta = json.loads(p.read_text())
                meta["timing"] = self.timing()
                p.write_text(json.dumps(meta, indent=1))
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
    """The policy cameras for this rig's render version: Filament in the Isaac scene (4) or Newton (1)."""
    if render_version(rig) >= RENDER_VERSION:
        return FilamentCameras(rig, **layout)
    return PolicyCameras(rig, **layout)
