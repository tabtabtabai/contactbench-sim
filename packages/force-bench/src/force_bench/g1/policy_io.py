"""The G1 + Dex1's state and action in our Isaac G1 + Dex1 harness's convention (matrix sims/isaac/run_episode.py and
third_party/unibot_submission), so one policy server runs unchanged on the Isaac tasks and on ContactBench.

State (observation.state.*):
  left_arm, right_arm   (7,) rad: shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw
  left_gripper, ...     (1,) the RAW Dex1 position: about 0 closed, 4.3 open (< 3.0 counts as closed)
  lower_body            (15,) legs (12) then waist yaw, roll, pitch: frozen here, the eval stance
  {side}_ee_pose_gripper_base  (6,) the TCP (dex1_base_link + 63.5 mm along +x) in the pelvis frame: xyz (m),
                        rpy (extrinsic xyz, as pinocchio's matrixToRpy), the convention sims/common/fk.py validates
                        against the datasets
Action (action.*): left_arm / right_arm (7,) absolute joint targets, rad; left_gripper / right_gripper (1,) absolute
raw Dex1 targets.

Raw gripper <-> finger joint: raw = clip(5.4 * (0.024 - q) / 0.044, 0, 5.4), the Isaac sim's own DDS conversion
(matrix sims/isaac/paper_cup_stacking/dataset/release/pc_training_export.py:43; it agrees with run_episode.py's
4.3-open / 3.0-closed thresholds).  Our Dex1 finger joints run -0.02 (83 mm open) .. 0.0245 (closed), the same
convention (+0.024 closed).
"""

from __future__ import annotations

import numpy as np

from force_bench.g1.robot import PREFIX

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
RAW_MAX, Q_CLOSED_REF, STROKE_M = 5.4, 0.024, 0.044
TCP_OFFSET = np.array([0.0635, 0.0, 0.0])          # dex1_base_link -> the gripper point (sims/common/fk.py)
# the frozen lower body, in the dataset's order: 12 leg joints (the preset's 0) then the eval waist (yaw, roll,
# pitch) = sims/isaac/eval_config_defaults.json init_waist, which the exported model's frozen waist matches
LOWER_BODY = np.r_[np.zeros(12), -0.002719, 0.00738, 0.184662].astype(np.float32)


def raw_of_q(q: float) -> float:
    return float(np.clip(RAW_MAX * (Q_CLOSED_REF - q) / STROKE_M, 0.0, RAW_MAX))


def q_of_raw(raw: float) -> float:
    return float(Q_CLOSED_REF - np.clip(raw, 0.0, RAW_MAX) * STROKE_M / RAW_MAX)


def _rot_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_of(R) -> np.ndarray:
    """Extrinsic-xyz roll, pitch, yaw (R = Rz(yaw) Ry(pitch) Rx(roll)), pinocchio's matrixToRpy."""
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])),
                     np.arctan2(R[1, 0], R[0, 0])])


class G1IO:
    """Index maps over a finalized G1 rig: read the state, write a row of targets."""

    def __init__(self, model):
        labels = [str(x or "") for x in model.joint_label]
        # three layouts: joint_q (coords), joint_target_q (the targets: the coord layout, the rows' tq_index) and
        # joint_f / effort limits (dofs, joint_qd_start); a free joint makes them differ
        qs, ds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        ts = model.joint_target_q_start.numpy()
        idx = {lb: j for j, lb in enumerate(labels)}
        self.model = model
        self.arm_q, self.arm_t, self.arm_d, self.arm_j, self.fin_q, self.fin_t = {}, {}, {}, {}, {}, {}
        for side in ("left", "right"):
            js = [idx[f"{PREFIX}{side}_{n}_joint"] for n in ARM]
            self.arm_j[side] = js
            self.arm_q[side] = np.array([int(qs[j]) for j in js])
            self.arm_t[side] = np.array([int(ts[j]) for j in js])
            self.arm_d[side] = np.array([int(ds[j]) for j in js])
            fs = [idx[f"{PREFIX}{side}_dex1_finger_joint_{k}"] for k in (1, 2)]
            self.fin_q[side] = np.array([int(qs[j]) for j in fs])
            self.fin_t[side] = np.array([int(ts[j]) for j in fs])
        # the gravity feed-forward each arm joint may add: its URDF cap less the actuator cap the row set (the
        # scripted rows reserve that headroom on the arm they use; an arm without it gets none, as in those rows)
        from force_bench.g1.robot import arm_cap
        lim = model.joint_effort_limit.numpy()
        self.ff_max = {side: np.array([max(0.0, (arm_cap(f"{side}_{n}_joint") or 0.0) - float(lim[d]))
                                       for n, d in zip(ARM, self.arm_d[side])]) for side in ("left", "right")}
        bl = [str(b) for b in model.body_label]
        self.pelvis = bl.index(f"{PREFIX}pelvis")
        self.gripper_base = {s: bl.index(f"{PREFIX}{s}_dex1_base_link") for s in ("left", "right")}
        self.lo = model.joint_limit_lower.numpy()
        self.hi = model.joint_limit_upper.numpy()

    def state(self, state) -> dict:
        q = state.joint_q.numpy()
        bq = state.body_q.numpy()
        P = bq[self.pelvis]
        Rp, pp = _rot_xyzw(P[3:7]), P[:3]
        out = {"observation.state.lower_body": LOWER_BODY.copy()}
        for side in ("left", "right"):
            out[f"observation.state.{side}_arm"] = q[self.arm_q[side]].astype(np.float32)
            out[f"observation.state.{side}_gripper"] = np.array(
                [raw_of_q(float(np.mean(q[self.fin_q[side]])))], dtype=np.float32)
            G = bq[self.gripper_base[side]]
            Rg = _rot_xyzw(G[3:7])
            tcp = G[:3] + Rg @ TCP_OFFSET
            out[f"observation.state.{side}_ee_pose_gripper_base"] = np.r_[
                Rp.T @ (tcp - pp), rpy_of(Rp.T @ Rg)].astype(np.float32)
        return out

    def apply(self, control, row: dict) -> None:
        """Absolute targets: the arms' 7 joints each and both Dex1 fingers from the raw gripper value."""
        tq = control.joint_target_q.numpy()
        for side in ("left", "right"):
            tq[self.arm_t[side]] = np.asarray(row[f"action.{side}_arm"], float).reshape(7)
            g = float(np.asarray(row[f"action.{side}_gripper"], float).reshape(-1)[0])
            tq[self.fin_t[side]] = q_of_raw(g)
        control.joint_target_q.assign(tq)

    def gravity_ff(self, state, control) -> None:
        """The robot's own gravity compensation on both arms from the live pose, every frame (the scripted rows feed
        the same torques forward), clipped at each joint's reserved headroom."""
        from force_bench.g1.robot import gravity_torques
        bq = state.body_q.numpy()
        jf = control.joint_f.numpy()
        for side in ("left", "right"):
            if self.ff_max[side].any():
                g = gravity_torques(self.model, bq, self.arm_j[side])
                jf[self.arm_d[side]] = np.clip(g, -self.ff_max[side], self.ff_max[side])
        control.joint_f.assign(jf)

    def target_row(self, control) -> dict:
        """The targets the robot is driven to now, as an action row (records a scripted expert for replay)."""
        tq = control.joint_target_q.numpy()
        out = {}
        for side in ("left", "right"):
            out[f"action.{side}_arm"] = tq[self.arm_t[side]].astype(np.float32)
            out[f"action.{side}_gripper"] = np.array([raw_of_q(float(np.mean(tq[self.fin_t[side]])))], np.float32)
        return out

    def current_row(self, state) -> dict:
        """The robot's current pose as an action row (holds it)."""
        s = self.state(state)
        return {f"action.{k.split('.')[-1]}": v for k, v in s.items() if k.split(".")[-1] in
                ("left_arm", "right_arm", "left_gripper", "right_gripper")}
