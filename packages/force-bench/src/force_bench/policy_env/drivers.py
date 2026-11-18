"""Drivers: who moves the arm.  A driver's ``advance(ep)`` steps the episode by one or more frames.

``ExpertDriver`` runs the row's scripted expert (privileged: it closes its loops on measured mechanism state, so it
is run in process and never served through an observation interface).  ``HoldDriver`` holds the start targets, the
null policy for wrapper checks.  The external-policy driver (real-time clock, matrix scene-observation requests)
lives in the matrix repo (``sims/contactbench``), next to the harness it talks to.
"""

from __future__ import annotations


class ExpertDriver:
    expert = True
    policy_id = "scripted-expert"
    privileged = True

    def begin(self, ep):
        pass

    def advance(self, ep):
        ep.step_expert()

    def close(self):
        pass


class HoldDriver:
    expert = False
    policy_id = "hold"
    privileged = False

    def begin(self, ep):
        self.q = ep.targets()

    def advance(self, ep):
        ep.step(self.q)

    def close(self):
        pass


class ThinkingExpertDriver(ExpertDriver):
    """The scripted expert with a simulated policy latency: every ``every_s`` of robot time it "thinks" for
    ``think_s``, its last targets held.  ``mode`` "realtime" steps the world through the thinking; "charged" settles
    first (Episode.charge_settled) and then freezes.  For the clock-mode check only (a moving policy, deterministic
    commands on robot time): the expert's own schedule is time-indexed, so its upstream verdict is not meaningful."""
    policy_id = "scripted-expert-thinking"

    def __init__(self, think_s=1.0, every_s=1.0, mode="realtime"):
        self.think_s, self.every_s, self.mode = float(think_s), float(every_s), mode
        self.next_think = self.every_s
        self.policy_id = f"scripted-expert-thinking-{self.think_s:g}s-every-{self.every_s:g}s"

    def advance(self, ep):
        if ep.robot_t + 1e-9 >= self.next_think:
            self.next_think += self.every_s + self.think_s
            ep.latency_s += self.think_s
            ep.n_requests += 1
            if self.mode == "charged":
                ep.charge_settled(self.think_s)
            else:
                for _ in range(int(round(self.think_s * ep.fps))):
                    if ep.done:
                        break
                    ep.step(None)
            return
        ep.step_expert()
