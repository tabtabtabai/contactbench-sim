"""The FR3 arm's joint torques against its limits, per frame: the contract's physics_qa.max_joint_torque_frac and
actuator_at_cap_frames, plus the effort caps READ BACK from the running model.

The G1 rows have had this since the uncapped-arm bug (g1-dex1 robot.TorqueMonitor); the FR3 rows had not, so a clean
FR3 pass meant "at-cap not measured".  Same method: each arm actuator's PD law from the state, targets and gains,
clipped at its cap, plus any joint feed-forward (control.joint_f); the cap is the model's joint_effort_limit (what the
MuJoCo backend maps onto actuator forcerange), cross-checked against the compiled MuJoCo model where it is reachable.
A cap of 0 or +inf on an arm joint is an UNCAPPED arm and is reported, not hidden."""
from __future__ import annotations

import numpy as np

ARM_DOF = 7
# Franka Research 3 rated joint torque limits (datasheet): joints 1-4 87 N.m, joints 5-7 12 N.m.
FR3_LIMIT_NM = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
FR3_LIMIT_SOURCE = "Franka Research 3 datasheet joint torque limits (87 N.m joints 1-4, 12 N.m joints 5-7)"


def _mujoco_caps(solver) -> list | None:
    """The compiled MuJoCo model's per-joint actuator force range for the first ARM_DOF joints, if reachable."""
    mj = getattr(solver, "mj_model", None)
    if mj is None:
        return None
    try:
        rng = np.asarray(mj.jnt_actfrcrange)[:ARM_DOF].astype(float)
        lim = np.asarray(mj.jnt_actfrclimited)[:ARM_DOF].astype(bool) if hasattr(mj, "jnt_actfrclimited") else None
        return [{"forcerange_Nm": [round(float(lo), 3), round(float(hi), 3)],
                 "limited": None if lim is None else bool(lim[k])} for k, (lo, hi) in enumerate(rng)]
    except Exception:                                  # a backend without the field: say so rather than guess
        return None


class ArmTorqueMonitor:
    def __init__(self, model, solver=None):
        self.ke = model.joint_target_ke.numpy()[:ARM_DOF].astype(float)
        self.kd = model.joint_target_kd.numpy()[:ARM_DOF].astype(float)
        self.cap = model.joint_effort_limit.numpy()[:ARM_DOF].astype(float)
        self.limit = np.array(FR3_LIMIT_NM)
        self.uncapped = [k for k, c in enumerate(self.cap) if not np.isfinite(c) or c <= 0.0]
        self.readback = {"joint_effort_limit_Nm": [round(float(c), 3) for c in self.cap],
                         "mujoco_jnt_actfrcrange": _mujoco_caps(solver),
                         "limit_Nm": list(FR3_LIMIT_NM), "limit_source": FR3_LIMIT_SOURCE,
                         "uncapped_joints": [f"fr3_joint{k + 1}" for k in self.uncapped]}
        self.peak = np.zeros(ARM_DOF)
        self.t_peak = np.zeros(ARM_DOF)
        self.sat = np.zeros(ARM_DOF, int)
        self.over = np.zeros(ARM_DOF, int)

    def sample(self, state, control, t):
        q = state.joint_q.numpy()[:ARM_DOF]
        qd = state.joint_qd.numpy()[:ARM_DOF]
        qt = control.joint_target_q.numpy()[:ARM_DOF]
        jf = getattr(control, "joint_f", None)
        ff = jf.numpy()[:ARM_DOF] if jf is not None else np.zeros(ARM_DOF)
        raw = self.ke * (qt - q) - self.kd * qd
        cap = np.where(np.isfinite(self.cap) & (self.cap > 0), self.cap, np.inf)
        act = np.clip(raw, -cap, cap)
        tot = np.abs(act + ff)
        self.sat += (np.abs(raw) >= cap - 1e-9).astype(int)
        self.over += (tot > self.limit + 1e-6).astype(int)
        m = tot > self.peak
        self.peak[m], self.t_peak[m] = tot[m], t

    def physics_qa(self) -> dict:
        names = [f"fr3_joint{k + 1}" for k in range(ARM_DOF)]
        frac = {n: round(float(self.peak[k] / self.limit[k]), 4) for k, n in enumerate(names)}
        at_cap = {n: int(self.sat[k]) for k, n in enumerate(names)}
        return {"max_joint_torque_frac": frac, "actuator_at_cap_frames": at_cap,
                "joint_torque_peak": {n: {"peak_Nm": round(float(self.peak[k]), 3), "t": round(float(self.t_peak[k]), 3),
                                          "frames_over_limit": int(self.over[k])} for k, n in enumerate(names)},
                "arm_effort_caps_sim_readback": self.readback,
                "joint_torque_note": "per arm joint over the episode: |PD actuator, clipped at its cap, + joint_f| "
                                     "against the FR3 datasheet limit"}

    def flags(self) -> list:
        out = []
        if (self.sat > 10).any():
            out.append("actuator_at_cap_over_10_frames")
        if self.uncapped:
            out.append("arm_uncapped")
        return out
