"""Build the G1 + Isaac Dex1 in a Newton ModelBuilder from its exported MuJoCo model (``g1_dex1_model.npz``).

Bodies, joint frames, axes, limits, inertia and meshes are MuJoCo's compiled model, verbatim.  The pelvis is welded to
the world at ``pelvis_xform`` (the whole-body controller's stance); legs and waist were frozen at export.  Shapes:
every MuJoCo geom becomes a visual mesh; only the Dex1 finger links collide, through boxes fitted to their meshes
(MuJoCo collides meshes as convex hulls; a box per link is the robust equivalent here).  Drives: MuJoCo's position
actuators' kp, a damping in proportion, and the actuators' force caps (the fingers' capped lower, see FINGER_EFFORT).
"""

from __future__ import annotations

import json

import numpy as np
import warp as wp

import newton

PREFIX = "g1/"
FINGER_EFFORT = 40.0          # N per finger: the squeeze (Dex1 spec 5-120 N)
COLLIDE_BODIES = ("hand_Link1_2", "hand_Link1_3", "hand_Link2_2", "hand_Link2_3")


def _xf(pos, quat_wxyz) -> wp.transform:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return wp.transform(wp.vec3(*[float(v) for v in pos]), wp.quat(x, y, z, w))


def _rot(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def add_g1(builder: newton.ModelBuilder, npz_path, pelvis_xform: wp.transform, mu: float = 1.0) -> dict:
    """Returns {"bodies": {mujoco name: newton body}, "joints": {mujoco joint name: newton joint},
    "q_index": {joint name: joint_q index}, "shapes": {body name: [shape ids]}}."""
    d = np.load(npz_path, allow_pickle=False)
    names = [str(n) for n in d["body_name"]]
    parent = d["body_parent"]
    drives = json.loads(str(d["drive_json"]))
    jnt_of = {int(b): j for j, b in enumerate(d["jnt_body"])}
    # world poses by forward composition (MuJoCo body_pos/quat are relative to the parent)
    Xw = {0: np.eye(4)}
    root = 1
    base = np.eye(4)
    p = wp.transform_get_translation(pelvis_xform)
    q = wp.transform_get_rotation(pelvis_xform)
    base[:3, :3] = _rot((q[3], q[0], q[1], q[2]))
    base[:3, 3] = [p[0], p[1], p[2]]
    for b in range(1, len(names)):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rot(d["body_quat"][b]), d["body_pos"][b]
        Xw[b] = (base @ T) if parent[b] == 0 else Xw[parent[b]] @ T
    out = {"bodies": {}, "joints": {}, "q_index": {}, "shapes": {}}
    joints = []
    for b in range(1, len(names)):
        R = _rot(d["body_iquat"][b])
        I = R @ np.diag(d["body_inertia"][b]) @ R.T
        X = Xw[b]
        from force_bench.policy_env.cameras import _quat_xyzw
        xq = _quat_xyzw(X[:3, :3])
        nb = builder.add_link(xform=wp.transform(wp.vec3(*X[:3, 3]), wp.quat(*xq)),
                              com=wp.vec3(*d["body_ipos"][b]), inertia=wp.mat33(*I.ravel()),
                              mass=float(max(d["body_mass"][b], 1e-4)), label=PREFIX + names[b])
        out["bodies"][names[b]] = nb
        pb = -1 if parent[b] == 0 else out["bodies"][names[parent[b]]]
        rel = _xf(d["body_pos"][b], d["body_quat"][b]) if parent[b] != 0 else pelvis_xform
        if b in jnt_of:
            j = jnt_of[b]
            jn = str(d["jnt_name"][j])
            jpos = d["jnt_pos"][j]
            pxf = wp.transform_multiply(rel, wp.transform(wp.vec3(*jpos), wp.quat_identity()))
            cxf = wp.transform(wp.vec3(*jpos), wp.quat_identity())
            kp, cap = drives.get(jn, (0.0, 0.0))
            finger = "finger" in jn
            kd = (0.01 if finger else 0.05) * kp
            lo, hi = (float(v) for v in d["jnt_range"][j])
            kw = dict(parent_xform=pxf, child_xform=cxf, axis=wp.vec3(*d["jnt_axis"][j]), limit_lower=lo,
                      limit_upper=hi, target_ke=kp, target_kd=kd,
                      effort_limit=(FINGER_EFFORT if finger else cap) or None,
                      armature=float(d["jnt_armature"][j]) if j < len(d["jnt_armature"]) else 0.0,
                      label=PREFIX + jn)
            if int(d["jnt_type"][j]) == 2:          # slide
                ji = builder.add_joint_prismatic(pb, nb, **kw)
            else:
                ji = builder.add_joint_revolute(pb, nb, **kw)
            out["joints"][jn] = ji
        else:
            ji = builder.add_joint_fixed(pb, nb, parent_xform=rel, label=PREFIX + names[b] + "_weld")
        joints.append(ji)
        # shapes
        sids = []
        collide = any(names[b].endswith(c) for c in COLLIDE_BODIES)
        verts_body = []
        for g in np.flatnonzero(d["geom_body"] == b):
            gx = _xf(d["geom_pos"][g], d["geom_quat"][g])
            rgba = d["geom_rgba"][g]
            if int(d["geom_type"][g]) == 7 and d["geom_mesh"][g] >= 0:       # mesh
                m = int(d["geom_mesh"][g])
                v, f = d[f"mesh{m}_v"], d[f"mesh{m}_f"]
                if d["geom_collide"][g] and collide:
                    Rg = _rot(d["geom_quat"][g])
                    verts_body.append(v @ Rg.T + d["geom_pos"][g])
                if rgba[3] <= 0.0:
                    continue                                                    # MuJoCo's invisible collision copy
                cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False,
                                                      density=0.0)
                sids.append(builder.add_shape_mesh(nb, xform=gx, mesh=newton.Mesh(v, f.ravel()), cfg=cfg,
                                                   color=wp.vec3(*[float(c) for c in rgba[:3]]),
                                                   label=f"{PREFIX}{names[b]}_vis{g}"))
        if collide and verts_body:
            V = np.vstack(verts_body)
            lo, hi = V.min(0), V.max(0)
            c = 0.5 * (lo + hi)
            h = 0.5 * (hi - lo)
            cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0, is_visible=False)
            sids.append(builder.add_shape_box(nb, xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                                              hx=float(h[0]), hy=float(h[1]), hz=float(h[2]), cfg=cfg,
                                              label=f"{PREFIX}{names[b]}_pad"))
        out["shapes"][names[b]] = sids
    builder.add_articulation(joints, label=PREFIX + "g1")
    return out


def q_indices(model: newton.Model, joints: dict) -> dict:
    qs = model.joint_q_start.numpy()
    return {n: int(qs[j]) for n, j in joints.items()}
