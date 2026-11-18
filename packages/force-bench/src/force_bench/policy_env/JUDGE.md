# World-state judges vs the rows' own verdicts

The rows' own verdicts are computed after the scripted expert's run, and some of their checks read the expert's
state or its schedule. The judges in `adapters.py` keep every threshold. They replace only those readings, with the
same quantity measured on the rollout. For expert runs both verdicts are computed on the same rollout, and they must
agree (clockwork-5 D3).

## 5.1 `turn_doorknob_open_door_flat`

| criterion | upstream | here |
|---|---|---|
| `locked_at_start` | the hidden lock probe (1.5 N·m at the hinge, 0.05–0.45 s) moves the leaf < 0.6° | identical; the env applies the probe whatever drives the arm |
| `unlocked_by_knob_then_opened` | the latch passes the jamb, the leaf moves < 1° before that, and the final angle is ≥ 5° | identical |
| `finger_grip_sustained` | per pad > 2 N in every 3-frame window from the expert's squeeze + 0.2 s (`T[2]+0.2`) to `T[-1]` | same gate, from latch clear to the episode end (the door is lifted through the knob); a policy that regrasps before unlocking is not failed for it, since it scores as a retry instead |
| `motion_stopped_door_held_open` | over the last 1 s, TCP drift < 3 mm, door range < 1°, final ≥ 5° | identical (the last 1 s of the episode) |
| `gear_coupling`, penetration, physics QA | integrity of the mechanism model | in the upstream verdict on the same rollout |

Online success signal: the latch has cleared, both pads are gripping, and the door has stayed ≥ 5° for the last
1 s with the TCP still. The episode ends when that holds, which is exactly when `motion_stopped_door_held_open` can
pass.

## 5.2 `install_light_bulb`

| criterion | upstream | here |
|---|---|---|
| `held_in_hand_at_start` | pads > 2 N over the first 0.5 s, drop < 1 mm | identical |
| `bulb_threaded_in` | the lead fitted from the expert's engagement detection to the seat is within 15% of CAD, and turns from first contact to tight ≥ 0.5 × revolutions | same thresholds; the fit starts a quarter turn past the first thread contact |
| `seated_and_tightened` | `tight_reason == "seat torque"` (the expert's own stop), then after release gap ≤ 0.3 mm, preload ≥ 1 N, tilt ≤ 2°, stalled | the expert's stop is replaced by the same condition measured on the rollout: seated, pad torque ≥ TAU_TIGHT with flank preload ≥ PRELOAD_TIGHT, then < 1° of turn over 0.15 s. The after-release thresholds are identical |
| `stationary_under_small_torque` | ±10 mN·m for 1 s each way after the hand has left; < 1° | identical; the probe window is found from the logged probe torque |
| `real_finger_grasp` | mean pads > 1 N over the expert's engage/twist strokes | mean pads > 1 N over the frames the hand is on the bulb while it turns tightening-way faster than 2°/s |
| `released_and_clear` | robot–bulb contact < 0.05 N from the end of the expert's retreat | same threshold, from the judge phase's start |

Judge phase (policy runs): the policy's part ends once it has let go for good, meaning no contact for 0.4 s and the
TCP 25 mm further from the bulb than where it held it, or when the budget runs out. The env then runs the row's own
ending sequence with the arm holding its last targets: still for 0.8 s, then the probe loosen-way for 1 s, then
tighten-way for 1 s. The expert runs its identical sequence itself.
