"""The back-out rule counts a setback only when the part stays > 30 deg past its best for 0.2 s (no GPU)."""
import numpy as np

from force_bench.policy_env.adapters import BACKOUT_HOLD_S, Persist, TurnTracker

DT = 1 / 60


def _run(turns_deg):
    tr, evs = TurnTracker(), []
    for i, d in enumerate(turns_deg):
        ev = tr.update(i * DT, d / 360)
        if ev:
            evs.append(ev)
    return evs


def test_a_one_frame_spike_is_not_a_setback():
    trace = np.r_[np.linspace(0, 970, 300), [970, 935, 970, 971, 972]]      # one frame 35 deg below best
    assert [e for e in _run(trace) if e[0] == "back_out"] == []


def test_the_5_7_rattle_is_not_a_setback():
    # 5.7 G1 nominal-0, 71.817-72.017 s: the nut's unwrapped tighten-way angle, deg (best 971.7 before it)
    rattle = [971.6, 952.9, 946.4, 963.7, 952.9, 937.4, 944.4, 958.8, 941.0, 951.5, 962.4, 966.6, 975.1]
    trace = np.r_[np.linspace(0, 971.7, 300), rattle]
    assert [e for e in _run(trace) if e[0] == "back_out"] == []


def test_a_sustained_back_out_is_a_setback_stamped_at_its_onset():
    n_hold = int(round(BACKOUT_HOLD_S / DT)) + 1
    trace = np.r_[np.linspace(0, 970, 300), np.full(n_hold + 5, 920.0), np.linspace(920, 1000, 30)]
    evs = _run(trace)
    backs = [e for e in evs if e[0] == "back_out"]
    assert len(backs) == 1 and abs(backs[0][1] - 300 * DT) < 1e-9          # onset = the first frame below
    assert [e[0] for e in evs] == ["back_out", "redo"]                    # progress again afterwards


def test_just_short_of_the_hold_is_not_a_setback():
    n_short = int(round(BACKOUT_HOLD_S / DT)) - 1
    trace = np.r_[np.linspace(0, 970, 300), np.full(n_short, 920.0), [970.0, 971.0]]
    assert [e for e in _run(trace) if e[0] == "back_out"] == []


def test_persist_fires_once_per_excursion():
    p = Persist(0.2)
    fired = [p.update(i * DT, True) for i in range(40)]
    assert sum(f is not None for f in fired) == 1
    p.update(1.0, False)
    assert p.update(1.0 + DT, True) is None and p.since is not None


def test_a_steady_recovery_closes_the_retry():
    # 2 deg per frame after the setback: no single-frame 10 deg jump, but 10 deg past the low point is a redo
    trace = np.r_[np.linspace(0, 970, 300), np.full(20, 920.0), 920 + 2.0 * np.arange(1, 10)]
    evs = _run(trace)
    assert [e[0] for e in evs] == ["back_out", "redo"]
