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
ARM_KP_SCALE = 10.0           # the arm servo, stiffer than matrix's MuJoCo actuators (kp 300/80): the open-loop
# scripted expert needs the hand within ~2 mm of its IK pose under gravity and contact (at 1x it sagged 15 mm and
# drifted 28 mm off the knob's axis while turning).  The joint effort caps stay the G1's (25 / 5 N.m); the real G1
# gets its tracking from the whole-body controller's gravity compensation instead.
COLLIDE_BODIES = ("hand_Link1_2", "hand_Link1_3", "hand_Link2_2", "hand_Link2_3")


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
            kp = kp if finger else kp * arm_kp_scale
            kd = (0.01 if finger else 0.04) * kp
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


class SumProbe:
    """One pad's force: the sum of several ForceProbes (a Dex1 finger is two colliding links)."""

    def __init__(self, probes):
        self.probes, self.history = probes, []

    def sample(self, contacts, state, t):
        rs = [p.sample(contacts, state, t) for p in self.probes]
        rec = {"t": t, "force": float(sum(r["force"] for r in rs)), "friction": float(sum(r["friction"] for r in rs))}
        self.history.append(rec)
        return rec


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
    return {
        "finger_effort_N": FINGER_EFFORT,
        "finger_effort_source": "the Dex1's rated grip, 5-120 N per finger (matrix sims/mujoco scene, Dex1 spec)",
        "pad_mu": PAD_MU,
        "pad_mu_source": ("matrix's MuJoCo Isaac Dex1 contact friction (sims/mujoco isaac_dex1.py: 'initial contact "
                          "parameters, not an identified PhysX material equivalence'); not measured on the real pads"
                          if PAD_MU == 1.0 else "NON-DEFAULT: a diagnostic value, not for published episodes"),
        "arm_kp_scale": ARM_KP_SCALE,
        "arm_kp_source": "10x matrix's MuJoCo arm actuator kp (300 shoulder/elbow, 80 wrist); effort caps unchanged "
                         "(25 / 5 N.m); stands in for the whole-body controller's tracking",
        "gravity_feedforward": True,
        "gravity_source": "the arm's gravity torques from the same MuJoCo model, fed forward (the whole-body "
                          "controller's gravity compensation)",
        "finger_collision": "a box per Dex1 finger link fitted to its collision mesh (MuJoCo collides meshes as hulls)",
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
