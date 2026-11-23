"""One CUDA graph per control frame: the frame's collision pass(es) and its solver substeps, captured once and replayed.

WHY.  Every scene here is ONE world, and a single-world MuJoCo Warp step is launch-bound, not compute-bound: the 5.1
G1 doorknob row (nv 23) issues about 4,800 kernel launches per 60 Hz frame (32 substeps x ~150), and the solver's
``capture_while`` loop, run outside a graph, copies its iteration counter to the host and synchronizes once per
solver iteration.  Measured on one L4 (contactbench:7724bb3, 5.1 seed 0 scripted expert, 23 Nov 2026):

    stepping                     ms / frame   host s / world s
    plain (Python launches)          299.2          17.95        with the policy cameras and the rollout recorder
    one graph per frame (this)        53.3           3.20
    plain, physics only              271.7          16.30        no cameras, recorder states only
    graph, physics only               28.2           1.69

py-spy put 74% of the plain run inside ``mujoco_warp.step`` on the host, a third of it packing kernel arguments.
Replaying a graph removes the Python launch cost and turns ``capture_while`` into a conditional graph node (no
readback), so the same kernels run in the same order with the same inputs.

WHAT IS CAPTURED.  Exactly the block ``Rig.run`` and ``Episode._physics`` run between the control callback and the
per-frame bookkeeping: ``clear_forces`` + ``pipeline.collide`` once per frame (or before every substep when the
scene collides every substep), and ``substeps`` x ``solver.step``.  Control targets, joint forces and model fields are
written IN PLACE between frames (``array.assign``), so the replay reads the current values.  The state double buffer
is swapped an even number of times per frame, so ``rig.s0`` / ``rig.s1`` name the same arrays after every frame.

WHEN IT STEPS PLAINLY INSTEAD (never a silent change of the physics):
  * ``TB_CUDA_GRAPH=0``, a CPU device, or a device without a CUDA memory pool;
  * an odd substep count, or a ``substep_hook`` (Python per substep);
  * a solver wrapper that is not declared graph safe (``graph_safe = True``), e.g. a wrapper that writes targets
    from Python every substep (kit ``SubstepTargets``);
  * the first ``WARMUP_FRAMES`` frames (kernels load and lazily created buffers are allocated outside a capture);
  * after a capture failed once on this rig (logged once).
A graph is recaptured whenever what it baked in changes: the substep count, the collision cadence, the solver, the
contact buffer or the state buffers, and after every ``SolverMuJoCo.notify_model_changed`` (which may rewrite or
reallocate solver arrays; the dial expert calls it when it changes its squeeze).

A graph-safe wrapper with host-side work does that work in ``pre_graph(control)`` before each replay, exactly where
its own ``step`` would have done it on the frame's first substep (``_kit_thread.ProbeTorque``).

DETERMINISM.  The default collision pipeline (``deterministic=False``) and the Newton->MuJoCo contact conversion
assign contact slots with atomics, so contact order depends on GPU thread scheduling and no run is bit-repeatable,
graph or not (measured: two plain 5.1 runs differ by up to 0.40 N of finger force from the first grasp contact).  The
graph changes nothing about that; see the PR for the outcome comparison on all five skill-5 tests.
"""
from __future__ import annotations

import os

import warp as wp

WARMUP_FRAMES = 3
_ENABLED = os.environ.get("TB_CUDA_GRAPH", "1") != "0"
_MAX_WRAPPERS = 8


def enabled() -> bool:
    return _ENABLED


# ── model-change generation: a captured graph is stale after notify_model_changed ──────────────────────────────────
_patched = False


def _install_notify_hook():
    """Count ``notify_model_changed`` calls per SolverMuJoCo instance (part of the graph key)."""
    global _patched
    if _patched:
        return
    import newton
    cls = newton.solvers.SolverMuJoCo
    orig = cls.notify_model_changed

    def notify_model_changed(self, *a, **k):
        r = orig(self, *a, **k)
        self._fb_model_generation = getattr(self, "_fb_model_generation", 0) + 1
        return r

    notify_model_changed.__wrapped__ = orig
    cls.notify_model_changed = notify_model_changed
    _patched = True


def _solver_chain(solver):
    """The solver and the solvers it wraps (``_solver``), outermost first."""
    chain, s = [], solver
    while s is not None and len(chain) < _MAX_WRAPPERS:
        chain.append(s)
        s = s.__dict__.get("_solver") if hasattr(s, "__dict__") else None
    return chain


def _innermost_mujoco(chain):
    import newton
    for s in chain:
        if isinstance(s, newton.solvers.SolverMuJoCo):
            return s
    return None


def graph_blocker(rig) -> str | None:
    """Why this rig's frame cannot be graph-captured right now (None: it can)."""
    if not _ENABLED:
        return "TB_CUDA_GRAPH=0"
    dev = rig.model.device
    if not dev.is_cuda:
        return "cpu device"
    if rig.substeps % 2:
        return f"odd substep count ({rig.substeps})"
    if getattr(rig, "substep_hook", None) is not None:
        return "substep_hook"
    chain = _solver_chain(rig.solver)
    mj = _innermost_mujoco(chain)
    if mj is None:
        return f"solver {type(rig.solver).__name__} is not SolverMuJoCo"
    if getattr(mj, "use_mujoco_cpu", False):
        return "use_mujoco_cpu"
    for s in chain:
        if s is not mj and not getattr(s, "graph_safe", False):
            return f"solver wrapper {type(s).__name__} is not graph safe"
    if not wp.is_mempool_enabled(dev):
        return "no CUDA memory pool"
    return None


class FrameStepper:
    """Steps one rig frame, through a cached CUDA graph when it can (see the module docstring)."""

    def __init__(self, rig):
        self.rig = rig
        self.graphs: dict = {}
        self.frames = 0
        self.failed: str | None = None
        self.reported: set = set()
        self.captures = 0
        _install_notify_hook()

    def _note(self, msg):
        if msg not in self.reported:
            self.reported.add(msg)
            print(f"[frame_graph] {self.rig.name}: {msg}", flush=True)

    def _block(self, every: bool, sim_dt: float):
        """The frame's physics exactly as the plain loop runs it (the substep hook only runs outside a graph)."""
        rig = self.rig
        hook = getattr(rig, "substep_hook", None)
        if not every:
            rig.s0.clear_forces()
            rig.pipeline.collide(rig.s0, rig.contacts)
        for _ in range(rig.substeps):
            if every:
                rig.s0.clear_forces()
                rig.pipeline.collide(rig.s0, rig.contacts)
            rig.solver.step(rig.s0, rig.s1, rig.control, rig.contacts, sim_dt)
            rig.s0, rig.s1 = rig.s1, rig.s0
            if hook is not None:
                hook()

    def step(self, sim_dt: float):
        """Advance the rig by one frame: ``substeps`` solver steps of ``sim_dt``."""
        rig = self.rig
        every = rig.effective_collide_every_substep()
        self.frames += 1
        why = self.failed or graph_blocker(rig)
        if why is not None or self.frames <= WARMUP_FRAMES:
            if why is not None:
                self._note(f"stepping without a CUDA graph: {why}")
            self._block(every, sim_dt)
            return
        chain = _solver_chain(rig.solver)
        for s in chain:                         # host-side work of graph-safe wrappers, before the replay
            pre = getattr(s, "pre_graph", None)
            if pre is not None:
                pre(rig.control)
        mj = _innermost_mujoco(chain)
        key = (every, rig.substeps, float(sim_dt), id(rig.solver), id(mj), getattr(mj, "_fb_model_generation", 0),
               id(rig.contacts), id(rig.pipeline), id(rig.s0), id(rig.s1), id(rig.control))
        g = self.graphs.get(key)
        if g is None:
            s0, s1 = rig.s0, rig.s1
            try:
                with wp.ScopedCapture(device=rig.model.device) as cap:
                    self._block(every, sim_dt)
            except Exception as e:  # noqa: BLE001 - fall back, never lose the run
                rig.s0, rig.s1 = s0, s1
                self.failed = f"capture failed ({type(e).__name__}: {str(e)[:200]})"
                self._note(f"stepping without a CUDA graph: {self.failed}")
                self._block(every, sim_dt)
                return
            if rig.s0 is not s0 or rig.s1 is not s1:          # an even swap count leaves the buffers in place
                rig.s0, rig.s1 = s0, s1
            self.graphs = {key: cap.graph}                   # older keys are stale (new solver, model change...)
            self.captures += 1
            if self.captures == 1:
                self._note(f"one CUDA graph per frame ({rig.substeps} substeps, "
                           f"{'collide every substep' if every else 'collide once per frame'})")
            g = cap.graph
        wp.capture_launch(g)                                  # a captured block does not run: launch it every frame


def stepper_for(rig) -> FrameStepper:
    st = getattr(rig, "_fb_frame_stepper", None)
    if st is None or st.rig is not rig:
        st = FrameStepper(rig)
        rig._fb_frame_stepper = st
    return st
