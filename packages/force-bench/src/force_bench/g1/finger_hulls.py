"""Write the Dex1 mesh-finger collision model: a CoACD convex decomposition of each finger link's collision mesh.

    uv run --with coacd --with trimesh --with scipy --with rtree python -m force_bench.g1.finger_hulls \\
        g1_dex1_model.npz dex1_finger_hulls.npz [--threshold 0.05]

The hulls are in each link's body frame (the geom pose applied), keyed ``<body>/<piece>/v|f``, with ``meta`` (JSON):
method, CoACD version, threshold, seed, the model's sha256 (add_g1 refuses a mismatch) and per link the piece count
and deviation against the source mesh, measured on 20k surface samples: ``mesh_outside_hulls_max_mm`` (mesh surface
not covered) and ``hull_off_mesh_max_mm`` / ``_p99_mm`` (hull surface standing off the mesh: bridged holes and
concavities), with the old fitted box's ``box_off_mesh_max_mm`` for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json

import numpy as np

LINKS = ("hand_Link1_2", "hand_Link1_3", "hand_Link2_2", "hand_Link2_3")


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _deviation_mm(trimesh, mesh, hulls, n=20000):
    pm, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    dm = np.full(len(pm), np.inf)
    for h in hulls:
        dist = np.abs(trimesh.proximity.signed_distance(h, pm))
        dm = np.minimum(dm, np.where(h.contains(pm), 0.0, dist))
    ph = np.vstack([trimesh.sample.sample_surface(h, max(200, n // len(hulls)), seed=0)[0] for h in hulls])
    _, dh, _ = trimesh.proximity.closest_point(mesh, ph)
    return 1000 * float(dm.max()), 1000 * float(dh.max()), 1000 * float(np.percentile(dh, 99))


def main():
    import coacd
    import trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="CoACD concavity threshold.  0.1: 11-12 pieces per finger body, 4.1 mm max stand-off; 0.05 "
                         "(47-52 pieces, 3.1 mm) overflowed MuJoCo Warp's 8000-contact buffer near the door knob")
    a = ap.parse_args()
    d = np.load(a.model, allow_pickle=False)
    names = [str(n) for n in d["body_name"]]
    res, arrays = {}, {}
    for b, n in enumerate(names):
        if not any(n.endswith(c) for c in LINKS):
            continue
        for g in np.flatnonzero(d["geom_body"] == b):
            if not d["geom_collide"][g] or d["geom_mesh"][g] < 0:
                continue
            m = int(d["geom_mesh"][g])
            v = d[f"mesh{m}_v"] @ _rot(d["geom_quat"][g]).T + d["geom_pos"][g]
            mesh = trimesh.Trimesh(v, d[f"mesh{m}_f"], process=True)     # welds the exported triangle soup
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()
            parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=a.threshold, seed=0,
                                    max_convex_hull=-1)
            hulls = [trimesh.Trimesh(pv, pf).convex_hull for pv, pf in parts]
            box = trimesh.creation.box(bounds=[v.min(0), v.max(0)])
            dmo, dho, dh99 = _deviation_mm(trimesh, mesh, hulls)
            _, bdh, _ = trimesh.proximity.closest_point(mesh, trimesh.sample.sample_surface(box, 20000, seed=0)[0])
            res[n] = {"mesh": m, "mesh_faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
                      "pieces": len(hulls), "hull_vertices_max": max(len(h.vertices) for h in hulls),
                      "mesh_outside_hulls_max_mm": round(dmo, 3), "hull_off_mesh_max_mm": round(dho, 3),
                      "hull_off_mesh_p99_mm": round(dh99, 3), "box_off_mesh_max_mm": round(1000 * float(bdh.max()), 3)}
            for i, h in enumerate(hulls):
                arrays[f"{n}/{i}/v"] = np.asarray(h.vertices, np.float32)
                arrays[f"{n}/{i}/f"] = np.asarray(h.faces, np.int32)
            print(n, json.dumps(res[n]), flush=True)
    # the pad plates' inner faces in the hand-base frame at finger joint 0 (every finger body's frame is a pure
    # translation of the hand base): the half-gap the experts' finger targets are measured from (DEX1_HALF_GAP0)
    def _offset(n):
        o = np.zeros(3)
        while not n.endswith("base_link"):
            b = names.index(n)
            o += d["body_pos"][b]
            n = names[int(d["body_parent"][b])]
        return o
    faces = {}
    for k, sgn in ((1, 1.0), (2, -1.0)):
        link = f"left_hand_Link{k}_3"
        V = np.vstack([arrays[x] for x in arrays if x.startswith(link + "/") and x.endswith("/v")]) + _offset(link)
        faces[k] = float(sgn * (V[:, 0].min() if sgn > 0 else V[:, 0].max()))
    if abs(faces[1] - faces[2]) > 1e-4:
        raise SystemExit(f"pad faces not symmetric: {faces}")
    meta = {"pad_face_half_gap_q0_m": round(0.5 * (faces[1] + faces[2]), 5),
            "pad_face_source": "the hand_Link{1,2}_3 hulls' inner faces in the hand-base frame at finger joint 0",
            "method": "CoACD", "coacd_version": importlib.metadata.version("coacd"), "threshold": a.threshold,
            "seed": 0, "model_sha256": hashlib.sha256(open(a.model, "rb").read()).hexdigest(), "links": res}
    np.savez(a.out, meta=json.dumps(meta), **arrays)


if __name__ == "__main__":
    main()
