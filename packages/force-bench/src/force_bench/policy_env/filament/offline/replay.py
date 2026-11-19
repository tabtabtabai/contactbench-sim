"""Replay a recorded ContactBench rollout through filament_server: per kept frame, set every body from rollout_states.npz
and render the requested cameras; write PNG frames per camera. Kinematic replay: physics is never run.

Cameras (any number, each "name:spec"):
  head:<body>         the Unitree G1 head camera mounted on <body> (g1/d435_link) with IsaacLab's camera_configs
                      g1_front_camera offset (rot (w,x,y,z) 0.4912,-0.50865,0.50865,-0.4912, "ros" convention), 10 mm
                      focal on a 20 mm aperture -> vertical FOV from --height/--width
  fixed:x,y,z,qx,qy,qz,qw,vfov   a fixed camera-to-world pose (GL axes) and vertical FOV in degrees
  third:<scene.json>  the "third" camera of an isaac_scene.py JSON (pose in GL axes, focal/aperture in mm)

usage: python replay.py --server filament_server --glb scene.glb --names bodies.txt --states rollout_states.npz
         --ibl DIR --out frames/ --size 640x480 --stride 2 --camera head:g1/d435_link [--camera fixed:...]
"""
import argparse
import struct
import subprocess
import time
from pathlib import Path

import numpy as np


def quat_mat(q):  # xyzw
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


ROS_ROT_WXYZ = (0.4912, -0.50865, 0.50865, -0.4912)          # camera_configs.g1_front_camera (built image)
ROS_TO_GL = np.diag([1.0, -1.0, -1.0])                       # ros optical (x right, y down, z fwd) -> GL (y up, -z fwd)


def _glb_node_topology(glb, name):
    """(uv [Nv,2], triangles [Nt,3]) of the named node's single primitive in a .glb."""
    import json
    raw = open(glb, "rb").read()
    jl = struct.unpack("<I", raw[12:16])[0]
    g = json.loads(raw[20:20 + jl])
    bin_ = raw[20 + jl + 8:]
    node = next(n for n in g["nodes"] if n.get("name") == name)
    prim = g["meshes"][node["mesh"]]["primitives"][0]

    def read(ai):
        acc = g["accessors"][ai]; bv = g["bufferViews"][acc["bufferView"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[acc["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
        o = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return np.frombuffer(bin_, dt, acc["count"] * nc, o).reshape(acc["count"], nc)
    return read(prim["attributes"]["TEXCOORD_0"]).astype(np.float32), read(prim["indices"]).reshape(-1, 3).astype(np.uint32)


def _vertex_normals(P, F):
    n = np.zeros_like(P)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True); p.add_argument("--glb", required=True)
    p.add_argument("--names", required=True); p.add_argument("--states", required=True)
    p.add_argument("--ibl"); p.add_argument("--ibl-none", action="store_true", help="pass - - (use --opt ibl_color)"); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", default="640x480"); p.add_argument("--stride", type=int, default=2)
    p.add_argument("--camera", action="append", required=True); p.add_argument("--lux", default="18000,75000")
    p.add_argument("--deform-points", help="[T, Nv, 3] float32 sidecar (world metres) of the glb's deform node")
    p.add_argument("--deform-normals", help="[T, Nv, 3] float32 normals sidecar (optional: computed if absent)")
    p.add_argument("--deform-node", default="deform/towel")
    p.add_argument("--opt", action="append", default=[], help="server look option key=value (repeatable)")
    p.add_argument("--head-focal-mm", type=float, default=7.6, help="7.6 = harness/eval head (spec 5.2)")
    p.add_argument("--head-extra-pitch-deg", type=float, default=0.0, help="extra tilt about the camera x axis (+ = down)")
    p.add_argument("--head-rot-ros", default=",".join(map(str, ROS_ROT_WXYZ)), help="w,x,y,z in the mount link (ros)"); p.add_argument("--head-aperture-mm", type=float, default=20.0)
    a = p.parse_args()
    W, H = map(int, a.size.split("x"))
    R = np.load(a.states, allow_pickle=True)
    labels = [str(x) for x in R["body_labels"]]
    Q = R["body_q"].astype(np.float32)
    w_, x_, y_, z_ = map(float, a.head_rot_ros.split(","))
    R_off = quat_mat((x_, y_, z_, w_))                       # camera (ros) in the mount link
    e = np.radians(a.head_extra_pitch_deg)                   # ros optical: +y down, so +angle about +x tilts down
    R_off = R_off @ np.array([[1, 0, 0], [0, np.cos(e), np.sin(e)], [0, -np.sin(e), np.cos(e)]])
    head_vfov = float(np.degrees(2 * np.arctan((a.head_aperture_mm * H / W) / 2 / a.head_focal_mm)))
    cams = []
    for spec in a.camera:
        kind, arg = spec.split(":", 1)
        if kind == "head":
            cams.append(("head", labels.index(arg), head_vfov))
        elif kind == "mount":
            # mount:<body>:px,py,pz:qw,qx,qy,qz:focal_mm[:aperture_mm]  a camera on a body, ros convention (IsaacLab)
            body, pos, rot, foc, *ap = arg.split(":")
            w2, x2, y2, z2 = map(float, rot.split(","))
            Rm = quat_mat((x2, y2, z2, w2)) @ ROS_TO_GL
            apm = float(ap[0]) if ap else 20.0
            vf = float(np.degrees(2 * np.arctan((apm * H / W) / 2 / float(foc))))
            cams.append(("mount", (labels.index(body), np.array(list(map(float, pos.split(","))), float), Rm), vf))
        elif kind == "third":
            import json
            t = json.loads(Path(arg).read_text())["third"]
            M = np.eye(4); M[:3, :3] = quat_mat(t["quat"]); M[:3, 3] = t["pos"]
            vf = float(np.degrees(2 * np.arctan((t["aperture_mm"] * H / W) / 2 / t["focal_mm"])))
            cams.append(("fixed", M, vf))
        else:
            v = list(map(float, arg.split(",")))
            M = np.eye(4); M[:3, :3] = quat_mat(v[3:7]); M[:3, 3] = v[:3]
            cams.append(("fixed", M, v[7]))
    lux = a.lux.split(",")
    if a.ibl_none:
        ktx_arg, sh_arg = "-", "-"
    else:
        ibl = Path(a.ibl); ktx_arg, sh_arg = str(next(ibl.glob("*_ibl.ktx"))), str(ibl / "sh.txt")
    DP = DN = None
    if a.deform_points:
        import json as _j
        uv, idx = _glb_node_topology(a.glb, a.deform_node)
        nv = len(uv)
        DP = np.fromfile(a.deform_points, np.float32).reshape(-1, nv, 3)
        DN = np.fromfile(a.deform_normals, np.float32).reshape(-1, nv, 3) if a.deform_normals else None
        topo = a.out / "deform_topology.bin"
        a.out.mkdir(parents=True, exist_ok=True)
        with open(topo, "wb") as fh:
            fh.write(np.array([nv, len(idx)], np.int32).tobytes()); fh.write(uv.astype(np.float32).tobytes())
            fh.write(idx.astype(np.uint32).tobytes())
        a.opt = a.opt + [f"deform={topo}", f"deform_node={a.deform_node}"]
        print(f"[replay] deform {a.deform_node}: {DP.shape[0]} frames x {nv} vertices, {len(idx)} triangles", flush=True)
    proc = subprocess.Popen([a.server, a.glb, a.names, ktx_arg, sh_arg, f"{W}x{H}", str(len(cams)), *lux, *a.opt],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        line = proc.stderr.readline().decode(errors="replace")
        if "[filament]" in line:
            print(line.strip(), flush=True)
        if "ready" in line or not line:
            break
    for i, _ in enumerate(cams):
        (a.out / f"cam{i}").mkdir(parents=True, exist_ok=True)
    from PIL import Image
    times = []
    for k, f in enumerate(range(0, len(Q), a.stride)):
        blobs = []
        for kind, ref, vfov in cams:
            if kind == "head":
                q = Q[f, ref].astype(float)
                Mb = np.eye(4); Mb[:3, :3] = quat_mat(q[3:]); Mb[:3, 3] = q[:3]
                M = np.eye(4); M[:3, :3] = Mb[:3, :3] @ R_off @ ROS_TO_GL; M[:3, 3] = q[:3]
            elif kind == "mount":
                bi, pm, Rm = ref
                q = Q[f, bi].astype(float); Rb = quat_mat(q[3:])
                M = np.eye(4); M[:3, :3] = Rb @ Rm; M[:3, 3] = q[:3] + Rb @ pm
            else:
                M = ref
            blobs.append(np.r_[M.astype(np.float32).T.reshape(-1), np.float32(vfov)].astype(np.float32).tobytes())
        t0 = time.perf_counter()
        extra = b""
        if DP is not None:
            fd = min(f, len(DP) - 1)
            P = DP[fd]
            Nn = DN[fd] if DN is not None else _vertex_normals(P, idx)
            extra = np.ascontiguousarray(np.concatenate([P, Nn], 1), np.float32).tobytes()
        proc.stdin.write(struct.pack("<i", 0x46494C31) + Q[f].tobytes() + b"".join(blobs) + extra); proc.stdin.flush()
        raw = proc.stdout.read(W * H * 3 * len(cams))
        times.append(time.perf_counter() - t0)
        for i in range(len(cams)):
            Image.fromarray(np.frombuffer(raw[i * W * H * 3:(i + 1) * W * H * 3], np.uint8).reshape(H, W, 3)).save(
                a.out / f"cam{i}" / f"f{k:05d}.png")
    proc.stdin.write(struct.pack("<i", 0)); proc.stdin.flush(); proc.wait()
    t = np.array(times[5:]) * 1e3
    print(f"[replay] {len(times)} frames x {len(cams)} cameras, {np.median(t):.1f} ms/request median -> {a.out}")


if __name__ == "__main__":
    main()
