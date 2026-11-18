"""The FR3 + Franka Hand as a ``matrix.robot-model.v1`` calibration, exported from the model the sim runs.

Exported from the rig's IK model, which is the same URDF (``fr3_franka_hand.urdf``) at the same base, but with
the fixed links (link8, hand, hand_tcp) kept rather than collapsed into link7.  The chain runs from
``fr3/fr3_link0`` through the seven revolute joints, link8 and the hand to ``harness_mount``: a fixed frame on the
hand whose x is the approach axis (hand z) and y the finger travel (hand y), the harness's mount convention
(``closing_axis`` +Y).  ``gripper`` describes the Franka Hand for the harness's native-robot path (clockwork-5
option A); keys fixed in STATUS-contactbench.md.
"""

from __future__ import annotations

import numpy as np

MOUNT_IN_HAND = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # columns: mount x, y, z in hand
FINGER_MAX_M = 0.04          # per finger (URDF limit); the hand opens 0.08 m


def _wxyz(q_xyzw) -> list:
    x, y, z, w = (float(v) for v in q_xyzw)
    return [w, x, y, z]


def _quat_of_matrix(R) -> list:
    from force_bench.policy_env.cameras import _quat_xyzw
    return _wxyz(_quat_xyzw(R))


def _pose_mat(row) -> np.ndarray:
    from force_bench.policy_env.cameras import _mat_of
    return _mat_of(row)


def export(rig) -> dict:
    m = rig.ik.model
    labels = list(m.joint_label)
    body_labels = list(m.body_label)
    jtype = m.joint_type.numpy()
    child = m.joint_child.numpy()
    X_p, X_c = m.joint_X_p.numpy(), m.joint_X_c.numpy()
    axis = m.joint_axis.numpy()
    lo, hi = m.joint_limit_lower.numpy(), m.joint_limit_upper.numpy()
    qd_start = m.joint_qd_start.numpy()
    import newton
    revolute = int(newton.JointType.REVOLUTE)
    joints = []
    target = body_labels.index("fr3/fr3_hand")
    chain = []
    b = target
    parent = m.joint_parent.numpy()
    while True:
        j = int(np.flatnonzero(child == b)[0])
        chain.append(j)
        if parent[j] < 0 or body_labels[parent[j]] == "fr3/fr3_link0":
            break
        b = int(parent[j])
    for j in reversed(chain):
        entry = {"name": str(labels[j]),
                 "parent_frame": {"xyz": [float(v) for v in X_p[j][:3]], "wxyz": _wxyz(X_p[j][3:7])},
                 "child_frame": {"xyz": [float(v) for v in X_c[j][:3]], "wxyz": _wxyz(X_c[j][3:7])}}
        if int(jtype[j]) == revolute:
            d = int(qd_start[j])
            ax = np.asarray(axis[d], float)
            if not np.allclose(ax, [0, 0, 1]):
                raise ValueError(f"{labels[j]}: expected a +Z joint axis, got {ax}")
            entry.update(axis="Z", limits=[float(lo[d]), float(hi[d])])
        joints.append(entry)
    joints.append({"name": "harness_mount_axes",
                   "parent_frame": {"xyz": [0.0, 0.0, 0.0], "wxyz": _quat_of_matrix(MOUNT_IN_HAND)},
                   "child_frame": {"xyz": [0.0, 0.0, 0.0], "wxyz": [1.0, 0.0, 0.0, 0.0]}})
    hand = _pose_mat(rig.ik.body_pose("fr3/fr3_hand"))
    tcp = _pose_mat(rig.ik.body_pose("fr3/fr3_hand_tcp"))
    offset = float((np.linalg.inv(hand) @ tcp)[2, 3])     # along the approach axis, hand origin -> pad centre
    return {"schema": "matrix.robot-model.v1", "root": "fr3/fr3_link0", "tip": "harness_mount",
            "asset": "newton franka_emika_panda fr3_franka_hand.urdf (ContactBench)",
            "fingertip_offset_m": round(offset, 4), "closing_axis": 1, "joints": joints}


def gripper(rig) -> dict:
    """The Franka Hand for the harness's native path: a parallel prismatic hand, 0..0.08 m between the pads."""
    pad = pad_size(rig)
    return {"type": "parallel-prismatic", "max_gap_m": 2 * FINGER_MAX_M, "pad_size_m": pad,
            "command": "gripper 0 = open (0.08 m), 1 = closed; each finger target = 0.04 * (1 - g)"}


def pad_size(rig) -> list:
    """Contact extent of one fingertip (length along the approach axis, width across), from the finger's collision
    shapes in the sim model."""
    m = rig.model
    bi = rig.body_index("fr3/fr3_leftfinger")
    sb = m.shape_body.numpy()
    scale = m.shape_scale.numpy()
    sizes = [2 * np.asarray(scale[s], float) for s in np.flatnonzero(sb == bi)]
    if not sizes:
        return [0.0, 0.0]
    s = max(sizes, key=lambda v: float(np.prod(v[v > 0])) if (v > 0).any() else 0.0)
    return [round(float(s[2]), 4), round(float(s[0]), 4)]


def gripper_of(q_fingers) -> float:
    """Measured gripper closure for the observation: 0 open .. 1 closed."""
    return float(np.clip(1.0 - float(np.mean(q_fingers)) / FINGER_MAX_M, 0.0, 1.0))


def fingers_of(g: float) -> float:
    return float(FINGER_MAX_M * (1.0 - np.clip(g, 0.0, 1.0)))


def limits(rig) -> np.ndarray:
    m = rig.ik.model
    return np.column_stack([m.joint_limit_lower.numpy()[:7], m.joint_limit_upper.numpy()[:7]]).astype(float)
