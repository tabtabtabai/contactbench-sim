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
| `finger_grip_sustained` | per pad > 2 N in every 3-frame window from the expert's squeeze + 0.2 s (`T[2]+0.2`) to `T[-1]` | same gate, from the latch clearing, or the start of the final opening (the last frame the door was shut, < 1°) if that is later, to the episode end. The opening that counts must be lifted through the knob; an earlier opening that lost its grip costs attempts instead |
| `motion_stopped_door_held_open` | over the last 1 s, TCP drift < 3 mm, door range < 1°, final ≥ 5° | identical (the last 1 s of the episode) |
| `gear_coupling`, penetration, physics QA | integrity of the mechanism model | in the upstream verdict on the same rollout |

Online success signal: the judge would pass now. That means the latch has cleared, and both pads have passed the
grip gate without a break since the final opening began. It also means the door has stayed ≥ 5° for the last 1 s
with the TCP still. The episode ends when that holds. (Up to fork 274187c it checked the grip only at that instant: one
native-flagship episode ended on a one-finger hold the judge rightly failed.)

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

## Seeded starts

- **5.1:** `--seed N` translates the door within the paper's 0.15 × 0.15 m square (±75 mm in x and y), with no yaw.
  Upstream's `Hinge` and `door_angle` take the hinge along world +y, so a yawed door needs both rewritten and the
  expert re-verified. Yaw (±15°, clockwork-5 D3) is a follow-up. Seeded and nominal runs are reported separately.
- **5.2:** upstream's own `--seed`: the socket moves within the same square, with no yaw (the socket is round).

## Bad endings (solvability proof)

`TB_BAD_ENDING=<name>` (opt-in, off by default) makes a row's scripted expert end badly in one named way, so the
judge must fail it. These are deliberately failed experts, not policies; a name the row does not implement exits.
The judge output carries `bad_ending: <name>` when the row applied one.

| Row | Name | What the expert does instead |
|---|---|---|
| 5.1 doorknob | `forced_door` | never turns the knob (the latch stays in the strike) and pulls the leaf up the hinge arc on the reference, without the bounded lead |
| 5.5 door handle | `release_early` | turns the lever and lifts as usual, lets go at 3° (short of 5°) and rises clear; the flat leaf falls back |
| 5.2 bulb (and the other `_kit_thread` screw-in rows) | `untightened` | stops twisting at the seat touch, before the tightness condition, then releases, retreats and is probed without re-tightening |
| 5.6 dial | `short_10deg` | turns and trims to 10° short of the target, then releases and retreats; the judged target is unchanged |
| 5.7 nut | `early_release` | 3 of the 7 tighten-way strokes, then the plan's own open, lift, retreat and still |
