"""force_bench.frame_graph: when a frame may be graph-captured, and that the plain path is the loop it replaced (no GPU).

The CUDA path itself (capture, replay, recapture after notify_model_changed) is exercised by the skill-5 runs on an L4
recorded in the PR that added it; these tests pin the rules that decide between it and the plain loop.
"""
from types import SimpleNamespace

import newton

from force_bench import frame_graph as fg
from force_bench.appendix_tasks._kit_thread import ProbeTorque


class _Dev:
    def __init__(self, cuda):
        self.is_cuda = cuda


class _FakeMuJoCo(newton.solvers.SolverMuJoCo):
    """A SolverMuJoCo that records its steps instead of building MuJoCo (``__init__`` is skipped)."""

    def __init__(self, log):          # noqa: D107 - deliberately not calling super()
        self.log = log
        self.use_mujoco_cpu = False

    def step(self, s_in, s_out, control, contacts, dt):
        self.log.append(("step", s_in, s_out, dt))


class _State:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def clear_forces(self):
        self.log.append(("clear", self.name))


class _Pipeline:
    def __init__(self, log):
        self.log = log

    def collide(self, state, contacts):
        self.log.append(("collide", state.name))


def _rig(substeps=4, every=False, cuda=False, solver=None, hook=None):
    log = []
    rig = SimpleNamespace(name="t", substeps=substeps, model=SimpleNamespace(device=_Dev(cuda)),
                          s0=_State("a", log), s1=_State("b", log), control=SimpleNamespace(), contacts=object(),
                          pipeline=_Pipeline(log), substep_hook=hook)
    rig.solver = solver if solver is not None else _FakeMuJoCo(log)
    rig.effective_collide_every_substep = lambda: every
    return rig, log


def _old_loop(rig, sim_dt, log):
    """The loop Rig.run / Episode._physics ran before frame_graph, on the same fakes."""
    every = rig.effective_collide_every_substep()
    if not every:
        rig.s0.clear_forces()
        rig.pipeline.collide(rig.s0, rig.contacts)
    for _ in range(rig.substeps):
        if every:
            rig.s0.clear_forces()
            rig.pipeline.collide(rig.s0, rig.contacts)
        rig.solver.step(rig.s0, rig.s1, rig.control, rig.contacts, sim_dt)
        rig.s0, rig.s1 = rig.s1, rig.s0
        if rig.substep_hook is not None:
            rig.substep_hook()


def _names(log):
    return [(e[0],) + tuple(getattr(x, "name", x) for x in e[1:]) for e in log]


def test_the_plain_path_is_the_old_loop():
    for every in (False, True):
        for substeps in (4, 5):
            a, la = _rig(substeps=substeps, every=every)
            b, lb = _rig(substeps=substeps, every=every)
            fg.stepper_for(a).step(0.01)
            _old_loop(b, 0.01, lb)
            assert _names(la) == _names(lb)
            assert (a.s0.name, a.s1.name) == (b.s0.name, b.s1.name)


def test_the_substep_hook_still_runs_every_substep():
    calls = []
    rig, _ = _rig(substeps=4, hook=lambda: calls.append(1))
    fg.stepper_for(rig).step(0.01)
    assert len(calls) == 4


def test_blockers():
    assert fg.graph_blocker(_rig(cuda=False)[0]) == "cpu device"
    rig, _ = _rig(cuda=True, substeps=3)
    assert fg.graph_blocker(rig) == "odd substep count (3)"
    rig, _ = _rig(cuda=True, hook=lambda: None)
    assert fg.graph_blocker(rig) == "substep_hook"

    class Unsafe:
        def __init__(self, s):
            self._solver = s
    rig, log = _rig(cuda=True)
    rig.solver = Unsafe(_FakeMuJoCo(log))
    assert fg.graph_blocker(rig) == "solver wrapper Unsafe is not graph safe"
    rig, _ = _rig(cuda=True, solver=object())
    assert fg.graph_blocker(rig) == "solver object is not SolverMuJoCo"


def test_only_declared_wrappers_are_graph_safe():
    log = []
    inner = _FakeMuJoCo(log)
    probe = ProbeTorque(inner, dof=0)
    assert probe.graph_safe and hasattr(probe, "pre_graph")
    chain = fg._solver_chain(probe)
    assert chain == [probe, inner] and fg._innermost_mujoco(chain) is inner

    class Unsafe:
        def __init__(self, s):
            self._solver = s

        def __getattr__(self, name):           # delegates, like the kit wrappers
            return getattr(self._solver, name)

    u = Unsafe(inner)
    assert not getattr(u, "graph_safe", False)
    assert fg._solver_chain(u) == [u, inner]


def test_probe_torque_writes_once_per_change():
    writes = []

    class JF:
        def __init__(self):
            self.v = [0.0, 0.0]

        def numpy(self):
            return list(self.v)

        def assign(self, a):
            writes.append(list(a))
            self.v = list(a)

    control = SimpleNamespace(joint_f=JF())
    p = ProbeTorque(_FakeMuJoCo([]), dof=1)
    p.pre_graph(control)
    assert writes == []                         # unchanged probe: no host write
    p.probe = 0.5
    p.pre_graph(control)
    p.step(None, None, control, None, 0.01)     # already written: step() does not write again
    assert writes == [[0.0, 0.5]]


def test_the_switch(monkeypatch):
    monkeypatch.setattr(fg, "_ENABLED", False)
    assert fg.graph_blocker(_rig(cuda=True)[0]) == "TB_CUDA_GRAPH=0"
