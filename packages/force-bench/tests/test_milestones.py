"""Each test's three milestones on synthetic world-state traces (no GPU)."""
import numpy as np

from force_bench.policy_env.adapters import (BulbMilestones, DialMilestones, DoorMilestones, NutMilestones,
                                             milestones_unreached)

DT = 1 / 60


def _names(ms):
    return [m["name"] for m in ms.as_list()]


def test_door_milestones_in_order_and_held():
    ms = DoorMilestones()
    for i in range(600):                       # 10 s: grip from 1 s, latch clears at 4 s, door 0 -> 30 deg -> 0.5 deg
        t = i * DT
        f = 8.0 if t >= 1.0 else 0.0
        door = 0.0 if t < 5 else (min(30.0, (t - 5) * 20) if t < 8 else 0.5)
        ms.update(t, f, f, t >= 4.0, door)
    out = ms.as_list()
    assert _names(ms) == ["grasped", "latch_cleared", "door_open_1deg"]
    assert out[0]["reached"] and abs(out[0]["t"] - 1.0) < 2 * DT
    assert out[1]["reached"] and abs(out[1]["t"] - 4.0) < 2 * DT
    assert out[2]["reached"] and out[2]["held_at_end"] is False          # it shut back below 1 deg


def test_door_short_touch_and_forced_door_reach_nothing():
    ms = DoorMilestones()
    for i in range(300):
        t = i * DT
        f = 8.0 if 1.0 <= t < 1.3 else 0.0                                # 0.3 s of contact: not a grasp
        ms.update(t, f, f, False, 3.0)                                    # open but never unlatched
    assert [m["reached"] for m in ms.as_list()] == [False, False, False]


def test_bulb_milestones():
    ms = BulbMilestones()
    for i in range(600):
        t = i * DT
        f = 3.0 if t >= 1.0 else 0.0
        disp = 0.0 if t < 2.0 else 0.008
        turns = max(0.0, (t - 4.0) * 0.5)
        ms.update(t, f, f, 1.0, disp, t >= 4.0, turns, 3.0)
    out = ms.as_list()
    assert _names(ms) == ["grasped_lifted", "threads_engaged", "half_turns"]
    assert abs(out[0]["t"] - 2.0) < 2 * DT and abs(out[1]["t"] - 4.0) < 2 * DT
    assert abs(out[2]["t"] - 7.0) < 2 * DT                                # 1.5 of 3 turns at 0.5 turn/s


def test_bulb_moved_without_a_grip_is_not_lifted():
    ms = BulbMilestones()
    for i in range(300):
        ms.update(i * DT, 0.0, 5.0, 1.0, 0.02, False, 0.0, 3.0)          # one pad only
    assert ms.as_list()[0]["reached"] is False


def test_dial_milestones_from_a_nonzero_start():
    ms = DialMilestones()
    tol, target = np.radians(1.0), np.radians(90.0)
    for i in range(600):
        t = i * DT
        f = 5.0 if t >= 0.5 else 0.0
        ang = np.radians(30.0) + min(1.0, max(0.0, (t - 2.0) / 4.0)) * (target - np.radians(30.0) + np.radians(5.0))
        ms.update(t, f, f, ang, target, tol)
    out = ms.as_list()
    assert _names(ms) == ["grasped", "half_turn", "near_target"]
    assert out[0]["reached"] and out[1]["reached"]
    assert abs(out[1]["t"] - (2.0 + 4.0 * 30 / 65)) < 2 * DT             # half of the 60 deg to go
    assert out[2]["reached"] and out[2]["held_at_end"] is False           # overshot to 5 deg past, 2x tol = 2 deg


def test_nut_milestones_need_the_thread_not_a_spin():
    lead, z0, full = 0.001, 0.029, 7.02
    eng = lambda z: float(np.clip(0.027 - (z - 0.00351), 0.0, 0.00702) * 1e3)  # noqa: E731
    spin = NutMilestones(z0, lead, eng, full)
    for i in range(300):
        spin.update(i * DT, i * DT * 1.0, z0)                             # turning in place: never caught
    assert [m["reached"] for m in spin.as_list()] == [False, False, False]
    ms = NutMilestones(z0, lead, eng, full)
    for i in range(1200):
        turn = i * DT * 0.5
        ms.update(i * DT, turn, z0 - turn * lead)
    out = ms.as_list()
    assert _names(ms) == ["threads_caught", "one_turn", "half_engaged"]
    assert all(m["reached"] for m in out)
    assert out[0]["t"] < out[1]["t"]


def test_reach_miss_emits_three_unreached():
    for mod, names in (("turn_doorknob_open_door_flat_g1", ["grasped", "latch_cleared", "door_open_1deg"]),
                       ("install_light_bulb_g1", ["grasped_lifted", "threads_engaged", "half_turns"]),
                       ("turn_dial_g1_expert", ["grasped", "half_turn", "near_target"]),
                       ("screw_nut_onto_screw_g1", ["threads_caught", "one_turn", "half_engaged"])):
        ms = milestones_unreached(mod)
        assert [m["name"] for m in ms] == names and not any(m["reached"] for m in ms)
        assert all(m["t"] is None for m in ms)
