"""Rule A's per-finger force: the sum of each contact's NORMAL-force magnitude on the finger's shapes (contract #1773,
19 Nov: pressing, not the twist's friction couple).  The full-force magnitude sum is kept alongside.

The rows' own pad probes (``ForceProbe``) report ``|SensorContact.total_force|``, the NET vector over the body's
contacts, so contacts on opposite faces of a part cancel: a G1 finger box chattering through a 10 mm dial bar read
549 N net while its contacts summed 8.3 kN.  This sums ``|f . n|`` per contact (``n`` the contact normal) for every
contact between a finger's shapes and any other shape: opposite-face normals add, so chatter through a part still shows,
but the friction couple that turns a knob (tangential forces either side of the pad's centre, which cancel in the net
and add in a full-force sum) does not count as squeeze.  ``|f|`` per contact is summed too, as the full series."""
from __future__ import annotations

import numpy as np

FR3_FINGERS = ("fr3/fr3_leftfinger", "fr3/fr3_rightfinger")


class FingerMagnitudes:
    def __init__(self, model, fingers=FR3_FINGERS, limit_n: float = 150.0, run_frames: int = 4):
        sb = model.shape_body.numpy()
        blab = [str(b) for b in model.body_label]
        missing = [f for f in fingers if f not in blab]
        if missing:
            raise ValueError(f"finger bodies not in the model: {missing}")
        self.fingers = tuple(fingers)
        self.mask = [np.isin(sb, [blab.index(f)]) for f in fingers]
        self.limit, self.run_frames = float(limit_n), int(run_frames)
        self.series = []                                   # (t, f_0, f_1) per frame
        k = len(fingers)
        self.peak = [(0.0, None)] * k
        self.peak_full = [(0.0, None)] * k
        self.over, self.run, self.longest = [0] * k, [0] * k, [0] * k

    def sample(self, contacts, t) -> tuple:
        f = getattr(contacts, "force", None)
        if f is None:
            raise RuntimeError("contacts.force is not populated: the magnitude-sum finger force cannot be measured")
        n = int(contacts.rigid_contact_count.numpy()[0])
        out = [0.0] * len(self.fingers)
        full = [0.0] * len(self.fingers)
        if n > 0:
            n = min(n, len(contacts.rigid_contact_shape0))
            s0 = contacts.rigid_contact_shape0.numpy()[:n].astype(int)
            s1 = contacts.rigid_contact_shape1.numpy()[:n].astype(int)
            ok = (s0 >= 0) & (s1 >= 0)
            a, b = np.clip(s0, 0, None), np.clip(s1, 0, None)
            F = f.numpy()[:n][:, :3].astype(float)
            nrm = contacts.rigid_contact_normal.numpy()[:n].astype(float)
            mag = np.linalg.norm(F, axis=1)
            fn = np.abs((F * nrm).sum(axis=1))             # the normal component's magnitude
            for k, m in enumerate(self.mask):
                sel = ok & (m[a] ^ m[b])                   # one side on this finger, the other anything else
                out[k] = float(fn[sel].sum())
                full[k] = float(mag[sel].sum())
        for k, v in enumerate(out):
            if v > self.peak[k][0]:
                self.peak[k] = (v, round(float(t), 4))
            if v > self.limit:
                self.over[k] += 1
                self.run[k] += 1
                self.longest[k] = max(self.longest[k], self.run[k])
            else:
                self.run[k] = 0
        for k, v in enumerate(full):
            if v > self.peak_full[k][0]:
                self.peak_full[k] = (v, round(float(t), 4))
        self.series.append((float(t), *out, *full))        # t, normal per finger, full magnitude per finger
        return tuple(out)

    def summary(self) -> dict:
        per = {name: {"force_N": round(self.peak[k][0], 1), "t": self.peak[k][1], "frames_over_limit": self.over[k],
                      "longest_run": self.longest[k], "full_magnitude_peak_N": round(self.peak_full[k][0], 1)}
               for k, name in enumerate(self.fingers)}
        k = max(range(len(self.fingers)), key=lambda i: self.peak[i][0])
        return {"force_N": round(self.peak[k][0], 1), "finger": self.fingers[k], "t": self.peak[k][1],
                "frames_over_limit": max(self.over), "longest_run": max(self.longest), "per_finger": per,
                "measure": "sum of contact NORMAL-force magnitudes per finger body (contract #1773, 19 Nov)",
                "full_magnitude_peak_N": round(max(v[0] for v in self.peak_full), 1),
                "limit_N": self.limit, "run_frames": self.run_frames}

    def flags(self, single_frame_n: float = 500.0) -> list:
        return [f"single_frame_finger_contact_over_{single_frame_n:g}N:{name}"
                for k, name in enumerate(self.fingers) if self.peak[k][0] > single_frame_n]
