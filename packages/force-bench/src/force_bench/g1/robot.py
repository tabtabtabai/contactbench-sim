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
import os
FINGER_EFFORT = float(os.environ.get("G1_FINGER_EFFORT", 40.0))   # N per finger: the squeeze (Dex1 spec 5-120 N)
ARM_KP_SCALE = float(os.environ.get("G1_ARM_KP_SCALE", 1.0))   # the arm servo against matrix's MuJoCo actuators'
# kp (300 shoulder/elbow, 80 wrist).  1x by default, with the gravity torques fed forward; at most 3x (a stiffer arm
# turns any graze into an impact).  The joint effort caps stay the G1's (25 / 5 N.m).
ARM_KP_MAX = 3.0
# the G1's arm joint torque limits (URDF g1_29dof_*: <limit effort>), N.m: the exported MuJoCo model carries no
# force range for the arm (0 = unlimited), so they are applied here
ARM_EFFORT_NM = {"shoulder_pitch": 25.0, "shoulder_roll": 25.0, "shoulder_yaw": 25.0, "elbow": 25.0,
                 "wrist_roll": 25.0, "wrist_pitch": 5.0, "wrist_yaw": 5.0}


def cap_headroom(builder: newton.ModelBuilder, g1: dict, joint_names, tau_ff_max) -> dict:
    """Cap each arm actuator at its URDF limit less the joint's largest gravity feed-forward (|tau| over the plan or
    the row), so actuator + feed-forward never exceeds the G1's limit.  Returns {joint: {limit, ff_max, actuator}}."""
    out = {}
    for n, g in zip(joint_names, tau_ff_max):
        cap = arm_cap(n)
        if cap is None:
            continue
        d = int(builder.joint_qd_start[g1["joints"][n]])
        act = max(0.0, cap - abs(float(g)))
        builder.joint_effort_limit[d] = act
        out[n] = {"limit_Nm": cap, "ff_max_Nm": round(abs(float(g)), 3), "actuator_cap_Nm": round(act, 3)}
    return out


def arm_cap(joint_name: str) -> float | None:
    for k, v in ARM_EFFORT_NM.items():
        if joint_name.endswith(f"_{k}_joint"):
            return v
    return None
COLLIDE_BODIES = ("hand_Link1_2", "hand_Link1_3", "hand_Link2_2", "hand_Link2_3")
# How the Dex1 finger links collide: "box" (one box per link fitted to its collision mesh's body-frame bounds: the
# published G1 sets) or "mesh" (a CoACD convex decomposition of each link's collision mesh, from G1_FINGER_HULLS:
# a file written by force_bench.g1.finger_hulls).  Contact parameters (mu, solref/solimp, margins) are the same.
FINGER_COLLISION = os.environ.get("G1_FINGER_COLLISION", "box")
FINGER_HULLS = os.environ.get("G1_FINGER_HULLS", "")
FINGER_COLLISION_MODEL: dict = {}        # filled by add_g1 in "mesh" mode (method, pieces, deviation, sources)
if FINGER_COLLISION not in ("box", "mesh"):
    raise ValueError(f"G1_FINGER_COLLISION={FINGER_COLLISION!r}: box or mesh")
# How a grip is commanded: "position" (the rows' finger joint targets, clipped to the joint travel: the published
# sets) or "force" (the Dex1's grip-force setting: a close puts the target past any contact, at the travel's end plus
# 1.25 x set force / kp, so the effort cap -- the set force -- binds wherever the jaws meet the part; the joint limit
# stays the physical stop).  A variant: every task run under it commands its grips this way.
FINGER_COMMAND = os.environ.get("G1_FINGER_COMMAND", "position")
if FINGER_COMMAND not in ("position", "force"):
    raise ValueError(f"G1_FINGER_COMMAND={FINGER_COMMAND!r}: position or force")
DEX1_Q_CLOSED = 0.0245                    # the finger joints' closed end of travel (limit_upper)
# Arm damping: "legacy" (kd = 0.04 kp: the box-finger sets) or "model" (the compiled MuJoCo model's actuator kv, from
# G1_DRIVE_GAINS: a file g1_dex1_drive_gains.py exports beside the model npz, checked against its sha256)
ARM_DAMPING = os.environ.get("G1_ARM_DAMPING", "legacy")
DRIVE_GAINS = os.environ.get("G1_DRIVE_GAINS", "")
if ARM_DAMPING not in ("legacy", "model"):
    raise ValueError(f"G1_ARM_DAMPING={ARM_DAMPING!r}: legacy or model")
ARM_KD_USED: dict = {}                    # joint -> kd, filled by add_g1


def _model_kv(npz_path) -> dict:
    import hashlib
    if not DRIVE_GAINS:
        raise ValueError("G1_ARM_DAMPING=model needs G1_DRIVE_GAINS (g1_dex1_drive_gains.py output)")
    g = json.load(open(DRIVE_GAINS))
    sha = hashlib.sha256(open(npz_path, "rb").read()).hexdigest()
    if g["model_npz_sha256"] != sha:
        raise ValueError(f"{DRIVE_GAINS} was exported for model {g['model_npz_sha256']}, not {sha}")
    return {n: float(v["kv"]) for n, v in g["gains"].items()}


def force_grip_target(kp: float, set_force: float | None = None) -> float:
    """Force-commanded close: the finger target 1.25 x set_force / kp past the closed end of travel, so the PD
    saturates at its effort cap (= the set force) at any contact inside the travel."""
    return DEX1_Q_CLOSED + 1.25 * float(FINGER_EFFORT if set_force is None else set_force) / float(kp)


def robot_name() -> str:
    """The record's robot: each finger model / grip command is its own embodiment and must not pool with another."""
    tags = (["mesh fingers"] if FINGER_COLLISION == "mesh" else []) + (["force grip"] if FINGER_COMMAND == "force"
                                                                        else [])
    return "Unitree G1 + Dex1" + (f" ({', '.join(tags)})" if tags else "")


def _load_hulls(npz_path) -> dict:
    """{body name: [(V, F)]} from G1_FINGER_HULLS, checked against the model it was decomposed from."""
    import hashlib
    if not FINGER_HULLS:
        raise ValueError("G1_FINGER_COLLISION=mesh needs G1_FINGER_HULLS (force_bench.g1.finger_hulls output)")
    H = np.load(FINGER_HULLS, allow_pickle=False)
    meta = json.loads(str(H["meta"]))
    model_sha = hashlib.sha256(open(npz_path, "rb").read()).hexdigest()
    if meta.get("model_sha256") != model_sha:
        raise ValueError(f"{FINGER_HULLS} was decomposed from model {meta.get('model_sha256')}, not {model_sha}")
    out = {}
    for k in H.files:
        if k.endswith("/v"):
            body, i, _ = k.rsplit("/", 2)
            out.setdefault(body, []).append((int(i), H[k].astype(np.float32), H[f"{body}/{i}/f"].astype(np.int32)))
    links = meta["links"]
    FINGER_COLLISION_MODEL.clear()
    FINGER_COLLISION_MODEL.update({
        "method": f"CoACD {meta['coacd_version']} convex decomposition (threshold {meta['threshold']}, seed "
                  f"{meta['seed']}) of each Dex1 finger link's collision mesh, one convex hull shape per piece",
        "hulls_file_sha256": hashlib.sha256(open(FINGER_HULLS, "rb").read()).hexdigest(),
        "source_mesh": f"g1_dex1_model.npz (sha256 {model_sha}): the finger links' MuJoCo collision meshes "
                       "(geom_collide), from matrix sims/mujoco's G1 + Dex1 export",
        "pieces": {b: v["pieces"] for b, v in links.items()},
        "max_hull_off_mesh_mm": max(v["hull_off_mesh_max_mm"] for v in links.values()),
        "max_mesh_outside_hulls_mm": max(v["mesh_outside_hulls_max_mm"] for v in links.values()),
        "per_link": links,
        "contact_parameters": "unchanged from the box model (mu, solref/solimp, margins, finger drive)",
        "pad_face_half_gap_q0_mm": round(1e3 * float(meta["pad_face_half_gap_q0_m"]), 2),
        "box_face_half_gap_q0_mm": round(1e3 * DEX1_HALF_GAP0_BOX, 2),
        "finger_targets": ("re-derived from the hull pad faces: every expert's finger target is a half-gap measured "
                           "from DEX1_HALF_GAP0, the mesh pad plates' inner face at joint 0 (the box model used the "
                           "finger-body box's face, 3.5 mm proud of the real pad); no other change"),
    })
    return {b: [(v, f) for _, v, f in sorted(p, key=lambda t: t[0])] for b, p in out.items()}


def _xf(pos, quat_wxyz) -> wp.transform:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return wp.transform(wp.vec3(*[float(v) for v in pos]), wp.quat(x, y, z, w))


def _rot(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


PAD_MU = float(os.environ.get("G1_PAD_MU", 1.0))   # matrix's MuJoCo Dex1 contact mu; the kit's FR3 elastomer pads: 1.4


def add_g1(builder: newton.ModelBuilder, npz_path, pelvis_xform: wp.transform, mu: float = PAD_MU,
           arm_kp_scale: float = ARM_KP_SCALE) -> dict:
    """Returns {"bodies": {mujoco name: newton body}, "joints": {mujoco joint name: newton joint},
    "q_index": {joint name: joint_q index}, "shapes": {body name: [shape ids]}}."""
    d = np.load(npz_path, allow_pickle=False)
    names = [str(n) for n in d["body_name"]]
    parent = d["body_parent"]
    drives = json.loads(str(d["drive_json"]))
    jnt_of = {int(b): j for j, b in enumerate(d["jnt_body"])}
    hulls = _load_hulls(npz_path) if FINGER_COLLISION == "mesh" else {}
    model_kv = _model_kv(npz_path) if ARM_DAMPING == "model" else {}
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
    out = {"bodies": {}, "joints": {}, "q_index": {}, "shapes": {}, "colors": {}}
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
            if not finger and not cap:
                cap = arm_cap(jn) or 0.0               # the arm's URDF torque limit (the export has none)
            kp = kp if finger else kp * arm_kp_scale
            kd = (0.01 if finger else 0.04) * kp
            if not finger and ARM_DAMPING == "model" and kp > 0:
                kd = model_kv[jn]                # the compiled model's actuator kv (biasprm[2])
            if not finger and kp > 0:
                ARM_KD_USED[jn] = kd
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
                sid = builder.add_shape_mesh(nb, xform=gx, mesh=newton.Mesh(v, f.ravel()), cfg=cfg,
                                             color=wp.vec3(*[float(c) for c in rgba[:3]]),
                                             label=f"{PREFIX}{names[b]}_vis{g}")
                sids.append(sid)
                out["colors"][sid] = [float(c) for c in rgba[:3]]
        if collide and verts_body and FINGER_COLLISION == "mesh":
            cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0, is_visible=False)
            for i, (hv, hf) in enumerate(hulls[names[b]]):
                sids.append(builder.add_shape_convex_hull(nb, xform=wp.transform_identity(),
                                                          mesh=newton.Mesh(hv, hf.ravel()), cfg=cfg,
                                                          label=f"{PREFIX}{names[b]}_hull{i}"))
        elif collide and verts_body:
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
    for jn, ji in out["joints"].items():          # never an unlimited arm joint (the export once carried none)
        if arm_cap(jn) is not None:
            lim = float(builder.joint_effort_limit[int(builder.joint_qd_start[ji])])
            if not (0.0 < lim <= arm_cap(jn) + 1e-9):
                raise ValueError(f"{jn}: effort limit {lim} is not within the G1's {arm_cap(jn)} N.m")
    return out


def q_indices(model: newton.Model, joints: dict) -> dict:
    qs = model.joint_q_start.numpy()
    return {n: int(qs[j]) for n, j in joints.items()}


class SumProbe:
    """One pad's force: the sum of several ForceProbes (a Dex1 finger is two colliding links)."""

    def __init__(self, probes):
        self.probes, self.history = probes, []

    def sample(self, contacts, state, t):
        rs = [p.sample(contacts, state, t) for p in self.probes]
        rec = {"t": t, "force": float(sum(r["force"] for r in rs)), "friction": float(sum(r["friction"] for r in rs))}
        self.history.append(rec)
        return rec


def _qrot(q_xyzw, v):
    q, w = q_xyzw[..., :3], q_xyzw[..., 3:4]
    t = 2.0 * np.cross(q, v)
    return v + w * t + np.cross(q, t)


CURRENT_PHASE = [None]          # the row's phase this frame, as last given to TorqueMonitor.sample


class ContactPeak:
    """Which contact is behind a force: every frame, the robot's contacts (either shape on a ``prefix`` body) summed
    per shape pair; keeps the strongest pair each frame (``top``), the episode's peak pair and the frames whose
    strongest pair is above LOG_N.  Registered as a rig probe, so it is sampled with the pads every frame (its
    ``force`` is that frame's strongest robot pair)."""

    LOG_N = 50.0
    MAX_LOG = 10 ** 6           # every frame above LOG_N (the 300-frame cap truncated a 40 s handle episode)

    def __init__(self, model, prefix: str = PREFIX):
        self.shape_labels = [str(x or "") for x in model.shape_label]
        self.body_labels = [str(x or "") for x in model.body_label]
        self.shape_body = model.shape_body.numpy()
        self.robot = np.array([self._body(i).startswith(prefix) for i in range(len(self.shape_labels))], bool)
        self.history, self.log, self.top, self.peak, self.n_over = [], [], [], None, 0
        self.peak_normal = None     # the frame's strongest pair by its NORMAL sum (sum of |f.n|), the episode's peak

    def _body(self, shape):
        b = int(self.shape_body[shape])
        return self.body_labels[b] if b >= 0 else "world"

    def sample(self, contacts, state, t):
        rec = {"t": t, "force": 0.0, "friction": 0.0}
        self.top = []
        f = getattr(contacts, "force", None)
        n = int(contacts.rigid_contact_count.numpy()[0]) if f is not None else 0
        if n > 0:
            n = min(n, len(contacts.rigid_contact_shape0))
            s0 = contacts.rigid_contact_shape0.numpy()[:n].astype(int)
            s1 = contacts.rigid_contact_shape1.numpy()[:n].astype(int)
            ok = (s0 >= 0) & (s1 >= 0)
            rob = ok & (self.robot[np.clip(s0, 0, None)] | self.robot[np.clip(s1, 0, None)])
            if rob.any():
                idx = np.flatnonzero(rob)
                fv = f.numpy()[:n][idx][:, :3].astype(float)
                F = np.linalg.norm(fv, axis=1)
                Fn = np.abs((fv * contacts.rigid_contact_normal.numpy()[:n][idx][:, :3].astype(float)).sum(axis=1))
                pairs, normal = {}, {}
                for i, fi, fni in zip(idx, F, Fn):
                    a, b = int(s0[i]), int(s1[i])
                    if not self.robot[a]:
                        a, b = b, a                       # the robot's shape first
                    tot, best_i, best_f = pairs.get((a, b), (0.0, i, -1.0))
                    pairs[(a, b)] = (tot + fi, i if fi > best_f else best_i, max(best_f, fi))
                    normal[(a, b)] = normal.get((a, b), 0.0) + fni
                if normal:
                    (na, nb), nmax = max(normal.items(), key=lambda kv: kv[1])
                    if nmax > 1e-6 and (self.peak_normal is None or nmax > self.peak_normal["force_N"]):
                        self.peak_normal = {"t": round(float(t), 3), "force_N": round(float(nmax), 1),
                                            "full_N": round(float(pairs[(na, nb)][0]), 1),
                                            "shape_a": self.shape_labels[na], "shape_b": self.shape_labels[nb],
                                            "body_a": self._body(na), "body_b": self._body(nb),
                                            "phase": CURRENT_PHASE[0]}
                ranked = sorted(pairs.items(), key=lambda kv: -kv[1][0])[:3]
                if ranked and ranked[0][1][0] > 1e-6:
                    bq = state.body_q.numpy()
                    p0 = contacts.rigid_contact_point0.numpy()
                    for (a, b), (tot, i, _) in ranked:
                        sh = int(s0[i])
                        bi = int(self.shape_body[sh])
                        pt = p0[i].astype(float)
                        if bi >= 0:
                            pt = bq[bi, :3] + _qrot(bq[bi, 3:7], pt)
                        self.top.append({"force_N": round(float(tot), 1), "normal_N": round(float(normal[(a, b)]), 1),
                                         "shape_a": self.shape_labels[a],
                                         "shape_b": self.shape_labels[b], "body_a": self._body(a),
                                         "body_b": self._body(b), "point": [round(float(v), 4) for v in pt]})
                    rec["force"] = float(ranked[0][1][0])
        if self.top:
            if self.peak is None or self.top[0]["force_N"] > self.peak["force_N"]:
                self.peak = {"t": round(float(t), 3), **self.top[0], "phase": CURRENT_PHASE[0]}
            if self.top[0]["force_N"] > self.LOG_N:
                self.n_over += 1
                if len(self.log) < self.MAX_LOG:
                    self.log.append({"t": round(float(t), 3), "pairs": self.top})
        self.history.append(rec)
        return rec

    def summary(self) -> dict:
        return {"peak": self.peak, "peak_normal": self.peak_normal, "log_above_N": self.LOG_N, "frames_above": self.n_over,
                "frames_above_logged": self.log}


def camera_mounts(npz_path, side: str) -> list:
    """The G1's own cameras from the exported model, as PolicyCameras mounts [(name, body label, T_body_cam, fovy)]:
    the head camera (the d435 mount, Isaac's g1_front_camera; the left of the challenge's head pair) and the active
    hand's wrist camera.  MuJoCo cameras look along -z with +y up, the GL convention the renderer uses."""
    d = np.load(npz_path, allow_pickle=False)
    if "cam_name" not in d:
        return []
    names = [str(n) for n in d["cam_name"]]
    bodies = [str(n) for n in d["body_name"]]
    want = {"head": "cam_left_high", "wrist": f"cam_{side}_wrist"}
    out = []
    for label, cam in want.items():
        if cam not in names:
            continue
        i = names.index(cam)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rot(d["cam_quat"][i]), d["cam_pos"][i]
        out.append((label, PREFIX + bodies[int(d["cam_body"][i])], T, float(d["cam_fovy"][i])))
    return out


DEX1_RATED_GRIP_N = (5.0, 120.0)   # the Dex1's grip force range (matrix sims/mujoco scene: "Dex1 spec: 5-120 N grip")


def variant_params() -> dict:
    """Every physics and control value this embodiment sets that is not the robot model's own, with its source, for
    the episode record (``robot_variant_params``).  Published episodes must stay inside the real Dex1's ratings."""
    if not DEX1_RATED_GRIP_N[0] <= FINGER_EFFORT <= DEX1_RATED_GRIP_N[1]:
        raise ValueError(f"squeeze {FINGER_EFFORT} N is outside the Dex1's rated {DEX1_RATED_GRIP_N} N")
    if FINGER_COLLISION == "mesh" and not FINGER_COLLISION_MODEL:
        raise ValueError("mesh fingers: variant_params() before add_g1 loaded the hulls")
    if not 1.0 <= ARM_KP_SCALE <= ARM_KP_MAX:
        raise ValueError(f"arm kp scale {ARM_KP_SCALE} is outside 1-{ARM_KP_MAX:g}x matrix's")
    return {
        "finger_effort_N": FINGER_EFFORT,
        "finger_effort_source": "the Dex1's rated grip, 5-120 N per finger (matrix sims/mujoco scene, Dex1 spec)",
        "pad_mu": PAD_MU,
        "pad_mu_source": ("matrix's MuJoCo Isaac Dex1 contact friction (sims/mujoco isaac_dex1.py: 'initial contact "
                          "parameters, not an identified PhysX material equivalence'); not measured on the real pads"
                          if PAD_MU == 1.0 else "NON-DEFAULT: a diagnostic value, not for published episodes"),
        "arm_kp_scale": ARM_KP_SCALE,
        "arm_kp_source": (f"{ARM_KP_SCALE:g}x the export's kp: 120 shoulder, 161.2 elbow and wrist, the isaac profile's "
                          "droop-calibrated gains; effort caps unchanged (25 / 5 N.m); with the gravity feed-forward "
                          "it stands in for the whole-body controller's tracking"),
        "arm_kd": ({n.replace("_joint", ""): round(v, 4) for n, v in ARM_KD_USED.items()} if ARM_KD_USED else None),
        "arm_kd_source": ("the isaac profile actuator kv from the compiled model, matching Isaac and PR #1779; earlier "
                          "box sets used kd = 0.04 kp" if ARM_DAMPING == "model" else
                          "kd = 0.04 kp (the box-finger sets' damping; the compiled model's actuator kv is 3.0 "
                          "shoulder / 2.5 elbow and wrist)"),
        "arm_effort_caps_Nm": dict(ARM_EFFORT_NM),
        "arm_effort_caps_source": ("the G1 URDF (g1_29dof_*) joint <limit effort>.  The gravity feed-forward is a "
                                   "joint torque outside the PD actuator, so each actuator is capped at the limit "
                                   "less that joint's largest feed-forward over the plan (cap_headroom): the joint's "
                                   "total stays within the limit"),
        "gravity_feedforward": True,
        "gravity_source": "the arm's gravity torques from the same MuJoCo model, fed forward (the whole-body "
                          "controller's gravity compensation)",
        "finger_collision": ("a CoACD convex decomposition of each Dex1 finger link's collision mesh (mesh fingers)"
                             if FINGER_COLLISION == "mesh" else
                             "a box per Dex1 finger link fitted to its collision mesh (MuJoCo collides meshes as hulls)"),
        **({"finger_collision_model": dict(FINGER_COLLISION_MODEL)} if FINGER_COLLISION == "mesh" else {}),
        **({"finger_command": (f"grip-force: set {FINGER_EFFORT:g} N (Dex1 rated 5-120 N); position target past "
                               "contact so the effort cap binds; joint limits unchanged"),
            "finger_command_source": ("the Dex1 is commanded by a grip-force setting (its rated 5-120 N), not a "
                                      "finger position: on a thin part it closes until the set force is reached, "
                                      "wherever that is inside its travel"),
            "finger_command_why": ("on the mesh geometry a position target at the joint limit leaves 4.5 mm of "
                                   "overdrive on the dial's 10 mm bar, about 36 N at the Dex1's 8000 N/m instead of "
                                   "the set 70 N"),
            "finger_command_target": "joint limit_upper 0.0245 m + 1.25 x set force / kp"}
           if FINGER_COMMAND == "force" else {"finger_command": "position (the rows' finger joint targets)"}),
        "setup": "matrix's MuJoCo port of the Isaac G1 + Dex1 tasks: pelvis 0.793 m on straight legs, table top "
                 "0.8703 m (Isaac 0.8373 + the 3.3 cm pelvis difference), the ColorArrangement reset pose (both arms, "
                 "waist), the idle arm holding it",
        "start_pose": "the reset pose with the ACTIVE hand already raised to the hover height (table + 10 cm or more): "
                      "at the reset pose that hand sits inside the door frame's near edge when the part is ~50 cm out",
    }


# the G1's own look (visual only): the model's colours, which are the Unitree URDF's ("white" 0.7 grey on 29 visuals,
# "dark" 0.2 grey on 7) and the Isaac Dex1's (~0.015 black jaws); the recorder's studio palette overwrites them
def apply_look(model: newton.Model, g1: dict) -> None:
    """Restore every G1 shape's own colour after the recorder's palette (finalize).  kit_look leaves robot shapes
    alone, so this sticks."""
    cols = model.shape_color.numpy().copy()
    for sid, rgb in g1["colors"].items():
        cols[sid] = rgb
    model.shape_color.assign(np.asarray(cols, dtype=np.float32))


def floor_and_table(builder: newton.ModelBuilder, floor_z: float, table_xy, half_xy) -> None:
    """The G1 stands on a floor below the ContactBench table top (z = 0): move the ground plane down to the floor and
    put a solid table under the board, so the legs are on the floor (the rig's ground plane sits at the table top)."""
    for i, lbl in enumerate(builder.shape_label):
        if lbl and "ground" in str(lbl):
            X = builder.shape_transform[i]
            builder.shape_transform[i] = wp.transform(wp.vec3(X[0], X[1], X[2] + floor_z),
                                                      wp.quat(X[3], X[4], X[5], X[6]))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5)
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(table_xy[0], table_xy[1], 0.5 * (floor_z - 0.01)),
                                                 wp.quat_identity()),
                          hx=float(half_xy[0]), hy=float(half_xy[1]), hz=float(0.5 * (-0.01 - floor_z)),
                          cfg=cfg, label="table_base")


def init_pose(npz_path) -> dict:
    """The reset pose of both arms (joint name -> rad), from the exported model."""
    d = np.load(npz_path, allow_pickle=False)
    if "arm_init_json" not in d:
        return {}
    init = json.loads(str(d["arm_init_json"]))
    names = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    return {f"{side}_{n}_joint": float(v) for side, vals in init.items() for n, v in zip(names, vals)}


# ── online control for rows without a precomputed plan (the kit's screw expert) ──────────────────────────────────
PAD_CENTRE = np.array([0.0, 0.120, 0.0])     # the Dex1 pads' centre in the hand-base frame (95-145 mm span)
# the FR3 TCP frame (z = approach, y = finger travel) -> the Dex1 hand frame (x = closing, y = approach, z = pad
# width): R_dex1 = R_fr3_tcp @ TCP_TO_HAND
TCP_TO_HAND = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
DEX1_HALF_GAP0 = 0.0215                      # finger joint 0 = a 43 mm pad gap; -0.02 = 83 mm (half-gap 0.0215 - q)
# The grip face's half-gap at finger joint 0, which every expert's finger target is measured from (dex1_q, the knob's
# close).  Box fingers: the finger-body box's inner face (21.55 mm, which stands 3.5 mm proud of the real pad plate
# over the pad).  Mesh fingers: the pad plates' own inner face, measured on the hulls (finger_hulls writes it).
DEX1_HALF_GAP0_BOX = DEX1_HALF_GAP0
if FINGER_COLLISION == "mesh" and FINGER_HULLS:
    DEX1_HALF_GAP0 = float(json.loads(str(np.load(FINGER_HULLS, allow_pickle=False)["meta"]))["pad_face_half_gap_q0_m"])


def _R_xyzw(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def gravity_torques(model: newton.Model, body_q: np.ndarray, joints) -> np.ndarray:
    """The torque each listed (revolute) joint must make to hold its distal bodies against gravity, from the live
    body poses: the whole-body controller's gravity compensation, fed forward.  ``joints``: newton joint indices."""
    g = np.asarray(model.gravity.numpy()).reshape(-1)[:3] if hasattr(model, "gravity") else np.array([0, 0, -9.81])
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    Xp, axes, qd0 = model.joint_X_p.numpy(), model.joint_axis.numpy(), model.joint_qd_start.numpy()
    com, mass = model.body_com.numpy(), model.body_mass.numpy()
    kids = {}
    for j in range(len(child)):
        kids.setdefault(int(parent[j]), []).append(int(child[j]))
    out = np.zeros(len(joints))
    for k, j in enumerate(joints):
        pb = int(parent[j])
        Rp, pp = (_R_xyzw(body_q[pb, 3:7]), body_q[pb, :3]) if pb >= 0 else (np.eye(3), np.zeros(3))
        RX = _R_xyzw(Xp[j, 3:7])
        point = pp + Rp @ Xp[j, :3]
        axis = Rp @ RX @ axes[int(qd0[j])]
        stack, tau = [int(child[j])], 0.0
        while stack:
            b = stack.pop()
            stack += kids.get(b, [])
            c = body_q[b, :3] + _R_xyzw(body_q[b, 3:7]) @ com[b]
            tau += float(axis @ np.cross(c - point, mass[b] * g))
        out[k] = -tau
    return out


class G1IK:
    """Kinematic IK for one G1 arm on a private copy of the model (newton.ik), in the FR3 TCP convention the kit's
    screw expert speaks: the TCP is the Dex1 pad centre, its frame mapped by TCP_TO_HAND.  Every other joint is held
    at its start (joint-limit objective with lower = upper)."""

    LIMIT_MARGIN = {"elbow": 0.15, "wrist": 0.25}   # rad inside the arm's limits, as the offline planner keeps

    def __init__(self, npz_path, pelvis_xform, side: str, start_q: dict):
        import newton.ik as ik
        b = newton.ModelBuilder()
        g = add_g1(b, npz_path, pelvis_xform)
        qi = {n: int(b.joint_q_start[j]) for n, j in g["joints"].items()}
        for n, v in start_q.items():
            if n in qi:
                b.joint_q[qi[n]] = float(v)
        self.model = b.finalize()
        self.g1 = g                                      # the private model's {bodies, joints, ...}
        self.side = side
        self.hand = PREFIX + f"{side}_hand_base_link"
        self.ee = list(self.model.body_label).index(self.hand)
        self.arm = [n for n in g["joints"] if n.startswith(f"{side}_") and "dex1" not in n
                    and any(k in n for k in ("shoulder", "elbow", "wrist"))]
        self.arm_q = [qi[n] for n in self.arm]
        q0 = self.model.joint_q.numpy()
        lo, hi = self.model.joint_limit_lower.numpy().copy(), self.model.joint_limit_upper.numpy().copy()
        dof = self.model.joint_qd_start.numpy()
        armdof = {int(dof[g["joints"][n]]) for n in self.arm}
        qs = self.model.joint_q_start.numpy()
        for n in self.arm:                               # the planner's margins inside the elbow and wrist limits
            m = self.LIMIT_MARGIN["elbow"] if "elbow" in n else self.LIMIT_MARGIN["wrist"] if "wrist" in n else 0.0
            d0 = int(dof[g["joints"][n]])
            lo[d0], hi[d0] = lo[d0] + m, hi[d0] - m
        for j in range(self.model.joint_count):          # lock every non-arm dof at its start value
            d0 = int(dof[j])
            if d0 in armdof or int(qs[j + 1] if j + 1 < len(qs) else len(q0)) - int(qs[j]) != 1:
                continue
            lo[d0] = hi[d0] = q0[int(qs[j])]
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        x = state.body_q.numpy()[self.ee]
        self.pos = ik.IKObjectivePosition(link_index=self.ee, link_offset=wp.vec3(*PAD_CENTRE),
                                          target_positions=wp.array([wp.vec3(*x[:3])], dtype=wp.vec3))
        self.rot = ik.IKObjectiveRotation(link_index=self.ee, link_offset_rotation=wp.quat_identity(),
                                          target_rotations=wp.array([wp.vec4(*x[3:7])], dtype=wp.vec4))
        lim = ik.IKObjectiveJointLimit(joint_limit_lower=wp.array(lo, dtype=float),
                                       joint_limit_upper=wp.array(hi, dtype=float), weight=100.0)
        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.solver = ik.IKSolver(model=self.model, n_problems=1, objectives=[self.pos, self.rot, lim],
                                  lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)

    @staticmethod
    def hand_quat(tcp_quat_xyzw) -> np.ndarray:
        from force_bench.policy_env.cameras import _quat_xyzw
        return np.asarray(_quat_xyzw(_R_xyzw(tcp_quat_xyzw) @ TCP_TO_HAND), float)

    def solve(self, target_pos, target_quat, iterations: int = 16) -> np.ndarray:
        """Returns the private model's joint_q (all joints); the arm's are at ``self.arm_q``."""
        self.pos.set_target_position(0, wp.vec3(*np.asarray(target_pos, dtype=float)))
        self.rot.set_target_rotation(0, wp.vec4(*self.hand_quat(target_quat)))
        self.solver.step(self.joint_q, self.joint_q, iterations=iterations)
        return self.joint_q.numpy()[0].copy()

    def _state(self):
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        return state

    def tcp_pose(self) -> np.ndarray:
        x = self._state().body_q.numpy()[self.ee]
        return np.r_[x[:3] + _R_xyzw(x[3:7]) @ PAD_CENTRE, x[3:7]]

    def body_pose(self, label: str) -> np.ndarray:
        return self._state().body_q.numpy()[list(self.model.body_label).index(label)].copy()

    def residual(self, target_pos, target_quat):
        """(position error m, rotation error rad) of the current solution against a TCP target."""
        x = self.tcp_pose()
        R = _R_xyzw(x[3:7])
        Rt = _R_xyzw(target_quat) @ TCP_TO_HAND
        return (float(np.linalg.norm(x[:3] - np.asarray(target_pos))),
                float(np.arccos(np.clip((np.trace(R @ Rt.T) - 1) / 2, -1, 1))))


class TorqueMonitor:
    """Per arm joint, the torque it makes each frame (the PD actuator's law, clipped at its cap, plus the gravity
    feed-forward), estimated from the state, targets and gains: the episode's peak |torque| against the G1's limit,
    when and in which phase, and how many frames the actuator sat at its cap."""

    def __init__(self, model, g1: dict, joint_names, tq_index: dict):
        qs, ds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        self.names = list(joint_names)
        js = [g1["joints"][n] for n in self.names]
        self.qi = np.array([int(qs[j]) for j in js])
        self.di = np.array([int(ds[j]) for j in js])
        self.ti = np.array([int(tq_index[n]) for n in self.names])
        self.ke = model.joint_target_ke.numpy()[self.di].astype(float)
        self.kd = model.joint_target_kd.numpy()[self.di].astype(float)
        self.cap = model.joint_effort_limit.numpy()[self.di].astype(float)
        self.limit = np.array([arm_cap(n) or np.inf for n in self.names])
        n = len(self.names)
        self.peak, self.t_peak, self.ph_peak = np.zeros(n), np.zeros(n), [None] * n
        self.sat = np.zeros(n, int)
        self.sat_phase = {}
        self.over = np.zeros(n, int)

    def sample(self, state, control, t, phase=None):
        CURRENT_PHASE[0] = phase                       # read by ContactPeak (sampled after the step, same frame)
        q = state.joint_q.numpy()[self.qi]
        qd = state.joint_qd.numpy()[self.di]
        qt = control.joint_target_q.numpy()[self.ti]
        ff = control.joint_f.numpy()[self.di]
        raw = self.ke * (qt - q) - self.kd * qd
        act = np.clip(raw, -self.cap, self.cap)
        tot = np.abs(act + ff)
        self.last = {"torque_Nm": tot, "frac": tot / self.limit, "at_cap": np.abs(raw) >= self.cap - 1e-9}
        at = (np.abs(raw) >= self.cap - 1e-9)
        self.sat += at.astype(int)
        if at.any():                                   # where in the episode an actuator sits at its cap
            ph = str(phase)
            for k in np.flatnonzero(at):
                d = self.sat_phase.setdefault(self.names[k], {})
                d[ph] = d.get(ph, 0) + 1
        self.over += (tot > self.limit + 1e-6).astype(int)
        m = tot > self.peak
        self.peak[m], self.t_peak[m] = tot[m], t
        for k in np.flatnonzero(m):
            self.ph_peak[k] = phase

    def summary(self) -> dict:
        return {n: {"peak_Nm": round(float(self.peak[k]), 3), "limit_Nm": float(self.limit[k]),
                    "margin_Nm": round(float(self.limit[k] - self.peak[k]), 3), "t": round(float(self.t_peak[k]), 3),
                    "phase": self.ph_peak[k], "frames_actuator_at_cap": int(self.sat[k]),
                    "frames_at_cap_by_phase": self.sat_phase.get(n, {}),
                    "frames_over_limit": int(self.over[k])} for k, n in enumerate(self.names)}


def effective_arm_caps(model: newton.Model, g1: dict) -> dict:
    """The arm actuators' effort limits READ BACK from the finalized model (what the solver applies), against the G1's
    limits.  Raises if any arm joint's range is 0 or unbounded: a G1 episode does not start uncapped."""
    ds = model.joint_qd_start.numpy()
    lim = model.joint_effort_limit.numpy()
    out = {}
    for jn, ji in g1["joints"].items():
        cap = arm_cap(jn)
        if cap is None:
            continue
        v = float(lim[int(ds[ji])])
        if not (0.0 < v < 1e4) or v > cap + 1e-6:
            raise SystemExit(f"refusing to run: {jn} effort range {v} (the G1 limit is {cap} N.m)")
        out[jn] = {"limit_Nm": cap, "actuator_cap_Nm": round(v, 3)}
    return out


def sim_cap_readback(solver, g1: dict, budget: dict | None = None, tag: str = "[g1]") -> dict:
    """Each arm joint's actuator force range as the SIMULATOR holds it: MuJoCo's jnt_actfrcrange (the clamp on the
    joint's total actuator force, which newton sets from joint_effort_limit) read from the compiled mj_model and, when
    present, from the GPU mjw_model the step uses.  Logs one line per joint; raises if any range is missing, zero or
    unbounded, or wider than the G1's limit."""
    import mujoco
    s = solver
    for _ in range(4):                                # unwrap probes / wrappers to the SolverMuJoCo
        if hasattr(s, "mj_model"):
            break
        s = getattr(s, "solver", None) or getattr(s, "inner", None) or getattr(s, "_solver", None)
        if s is None:
            raise SystemExit("refusing to run: no MuJoCo model to read the arm's force ranges from")
    mjm = s.mj_model
    gpu = None
    mjw = getattr(s, "mjw_model", None)
    if mjw is not None and hasattr(mjw, "jnt_actfrcrange"):
        gpu = mjw.jnt_actfrcrange.numpy().reshape(-1, mjm.njnt, 2)[0] if mjw.jnt_actfrcrange.numpy().ndim == 3 \
            else mjw.jnt_actfrcrange.numpy()
    names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(mjm.njnt)]
    out = {}
    for jn in g1["joints"]:
        cap = arm_cap(jn)
        if cap is None:
            continue
        hits = [i for i, n in enumerate(names) if n == jn or n.endswith("/" + jn) or n.endswith(jn)]
        if len(hits) != 1:
            raise SystemExit(f"refusing to run: {jn} not found once in the MuJoCo model ({len(hits)} matches)")
        i = hits[0]
        lo, hi = (float(v) for v in mjm.jnt_actfrcrange[i])
        limited = bool(mjm.jnt_actfrclimited[i])
        g_lo, g_hi = (float(v) for v in gpu[i]) if gpu is not None else (lo, hi)
        if not limited or not (0.0 < hi < 1e4) or abs(lo + hi) > 1e-6 or hi > cap + 1e-6 or abs(g_hi - hi) > 1e-6:
            raise SystemExit(f"refusing to run: {jn} actuator force range limited={limited} [{lo}, {hi}] "
                             f"(gpu [{g_lo}, {g_hi}]), G1 limit {cap} N.m")
        ff = (budget or {}).get(jn, {}).get("ff_max_Nm")
        print(f"{tag} cap readback {jn}: forcerange [{lo:.3f}, {hi:.3f}] N.m (mujoco jnt_actfrcrange"
              f"{', gpu ' + format(g_hi, '.3f') if gpu is not None else ''}; URDF limit {cap:g}, ff headroom "
              f"{(cap - hi):.3f}{'' if ff is None else f', planned ff peak {ff:.3f}'})", flush=True)
        out[jn] = {"forcerange_Nm": [round(lo, 4), round(hi, 4)], "gpu_forcerange_Nm": [round(g_lo, 4), round(g_hi, 4)]
                   if gpu is not None else None, "limit_Nm": cap, "source": "mujoco jnt_actfrcrange (compiled model)"}
    return out


def finger_readback(solver, model, g1: dict, side: str, overdrive_m: float, tag: str = "[g1]") -> dict:
    """The Dex1 finger actuators as the simulator holds them (MuJoCo jnt_actfrcrange, the drive stiffness from the
    model) and the squeeze a set overdrive past contact commands (stiffness x overdrive, clamped by the range)."""
    import mujoco
    s = solver
    for _ in range(4):
        if hasattr(s, "mj_model"):
            break
        s = getattr(s, "solver", None) or getattr(s, "inner", None) or getattr(s, "_solver", None)
    mjm = s.mj_model
    names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(mjm.njnt)]
    ds = model.joint_qd_start.numpy()
    ke = model.joint_target_ke.numpy()
    out = {}
    for k in (1, 2):
        jn = f"{side}_dex1_finger_joint_{k}"
        i = [j for j, n in enumerate(names) if n.endswith(jn)][0]
        hi = float(mjm.jnt_actfrcrange[i][1])
        kp = float(ke[int(ds[g1["joints"][jn]])])
        sq = min(hi, kp * overdrive_m)
        print(f"{tag} finger readback {jn}: forcerange +-{hi:.1f} N (mujoco jnt_actfrcrange), kp {kp:.0f} N/m, "
              f"overdrive {overdrive_m * 1e3:.1f} mm -> commanded squeeze {sq:.1f} N", flush=True)
        out[jn] = {"forcerange_N": round(hi, 2), "kp_N_per_m": round(kp, 1), "overdrive_mm": round(overdrive_m * 1e3, 2),
                   "commanded_squeeze_N": round(sq, 2)}
    return out


class MagnitudeProbe:
    """One finger's contact force as the SUM OF CONTACT MAGNITUDES over its shapes (optionally against the given
    counterpart shapes): unlike a net (vector) total, contacts on opposite faces of a thin part do not cancel, so a
    finger box chattering through a bar shows its real load.  ForceProbe-compatible (sample / history)."""

    def __init__(self, model, body_labels, counterpart_shapes=None):
        sb = model.shape_body.numpy()
        blab = [str(b) for b in model.body_label]
        bodies = [blab.index(b) for b in body_labels]
        self.mine = np.isin(sb, bodies)
        labels = [str(x or "") for x in model.shape_label]
        robot = np.array([b >= 0 and blab[b].startswith(PREFIX) for b in sb], bool)
        self.other = (np.isin(np.arange(len(labels)), [labels.index(s) for s in counterpart_shapes])
                      if counterpart_shapes else ~robot)           # default: against every non-robot shape
        self.history = []

    def sample(self, contacts, state, t):
        """force: the sum over the finger's contacts of |f . n| (the NORMAL-force magnitudes, rule A, contract #1773
        of 19 Nov: opposite-face normals still add, a twist's friction couple does not count as squeeze);
        force_full: the sum of the full |f| (the earlier rule; force <= force_full)."""
        rec = {"t": t, "force": 0.0, "force_full": 0.0, "friction": 0.0}
        f = getattr(contacts, "force", None)
        n = int(contacts.rigid_contact_count.numpy()[0]) if f is not None else 0
        if n > 0:
            n = min(n, len(contacts.rigid_contact_shape0))
            s0 = contacts.rigid_contact_shape0.numpy()[:n].astype(int)
            s1 = contacts.rigid_contact_shape1.numpy()[:n].astype(int)
            ok = (s0 >= 0) & (s1 >= 0)
            a, b = np.clip(s0, 0, None), np.clip(s1, 0, None)
            sel = ok & ((self.mine[a] & self.other[b]) | (self.mine[b] & self.other[a]))
            if sel.any():
                fv = f.numpy()[:n][sel][:, :3].astype(float)
                nv = contacts.rigid_contact_normal.numpy()[:n][sel][:, :3].astype(float)
                rec["force"] = float(np.abs((fv * nv).sum(axis=1)).sum())
                rec["force_full"] = float(np.linalg.norm(fv, axis=1).sum())
        self.history.append(rec)
        return rec


def add_rule_a_probes(rig, side: str, net_probes=None) -> tuple:
    """Rule A's per-finger force (contract #1773): each Dex1 jaw (hand_Link{k}_2 + hand_Link{k}_3) as the sum of its
    contact magnitudes against every non-robot shape, registered as rig.probes["g1/finger{k}"] and rig.rule_a_probes
    (the adapters' force limit reads them).  ``net_probes``: the row's own net-vector finger probes, kept alongside
    (rig.net_probes) for comparison."""
    for k in (1, 2):
        rig.probes[f"g1/finger{k}"] = MagnitudeProbe(rig.model, [f"{PREFIX}{side}_hand_Link{k}_{j}" for j in (2, 3)])
        for j in (2, 3):          # per link, for diagnosis only (Link_3 the pad plate, Link_2 the finger body)
            rig.probes[f"g1/finger{k}_link{j}"] = MagnitudeProbe(rig.model, [f"{PREFIX}{side}_hand_Link{k}_{j}"])
    rig.rule_a_probes = ("g1/finger1", "g1/finger2")
    rig.net_probes = tuple(net_probes or ())
    return rig.rule_a_probes
