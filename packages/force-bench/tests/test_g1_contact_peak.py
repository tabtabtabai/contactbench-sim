"""ContactPeak names the shape pair behind a robot contact force (the G1 + Dex1 rows' force_limit diagnosis)."""
import numpy as np
import pytest

pytest.importorskip("newton")
from force_bench.g1.robot import ContactPeak  # noqa: E402


class _A:
    def __init__(self, x):
        self.x = np.asarray(x)

    def numpy(self):
        return self.x

    def __len__(self):
        return len(self.x)


class _Model:
    shape_label = ["g1/right_hand_Link1_2/box", "door/frame", "g1/right_hand_Link2_2/box", "ground"]
    body_label = ["g1/right_hand_Link1_2", "door/frame", "g1/right_hand_Link2_2"]
    shape_body = _A([0, 1, 2, -1])


def _contacts(s0, s1, f):
    n = len(s0)
    c = type("C", (), {})()
    c.rigid_contact_count = _A([n])
    c.rigid_contact_shape0, c.rigid_contact_shape1 = _A(s0 + [-1] * 4), _A(s1 + [-1] * 4)
    c.force = _A([[*v, 0, 0, 0] for v in f] + [[0] * 6] * 4)
    c.rigid_contact_point0 = _A([[0.0, 0.0, 0.01]] * (n + 4))
    return c


def _state():
    s = type("S", (), {})()
    s.body_q = _A([[0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]])
    return s


def test_names_the_pair_and_keeps_the_peak():
    cp = ContactPeak(_Model())
    # the door frame on the left finger (two contacts, 120 + 80 N) beats the right finger on the ground (30 N);
    # the frame/ground contact (no robot shape) is ignored
    r = cp.sample(_contacts([1, 0, 2, 1], [0, 1, 3, 3], [[0, 0, 120], [80, 0, 0], [0, 30, 0], [0, 0, 999]]),
                  _state(), 1.5)
    assert r["force"] == pytest.approx(200.0)
    top = cp.top[0]
    assert (top["shape_a"], top["shape_b"]) == ("g1/right_hand_Link1_2/box", "door/frame")
    assert top["body_b"] == "door/frame" and top["point"] == [1.0, 0.0, 0.01]
    cp.sample(_contacts([2], [3], [[0, 0, 60]]), _state(), 1.6)
    assert cp.peak["force_N"] == 200.0 and cp.peak["t"] == 1.5 and cp.top[0]["body_b"] == "world"
    s = cp.summary()
    assert s["frames_above"] == 2 and len(s["frames_above_logged"]) == 2


def test_no_contacts():
    cp = ContactPeak(_Model())
    assert cp.sample(_contacts([], [], []), _state(), 0.0)["force"] == 0.0
    assert cp.peak is None and cp.top == []
