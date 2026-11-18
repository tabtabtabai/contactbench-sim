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
