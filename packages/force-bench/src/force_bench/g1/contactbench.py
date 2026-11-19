"""The Unitree G1 + Dex1 on matrix's ContactBench runner (sims/contactbench/runner.py --robot g1, its default robot).

Import-light on purpose: ``env_defaults`` must run before any other force_bench module is imported, because the G1
modules read their settings (finger model, arm damping, table, dial collider, the dial mode) from the environment at
import.  The other helpers import what they need when called.

Per test the G1 row, its settings and its plan (``ROWS``):
  5.1 knob and 5.5 handle: the offline MuJoCo plans at the fixed stance (data/plans), finger effort 80 N;
  5.2 bulb and 5.7 nut: the kit's own expert through the online IK shim, at their fixed stances;
  5.6 dial: the single-grasp expert (the FR3 row's own, through the G1 IK shim) at the one stance (0.27, 0.25), on the
      bar's exact box collider; only some seeds reach there (the others are full reach-miss records).  The two-stroke
      regrasp mode (G1_DIAL_REGRASP=1, stance (0.33, 0.10), with data/plans' IK tables) stays selectable.
Every G1 episode runs in the Isaac challenge scene (TB_TABLE=isaac), with mesh fingers and the model's arm kv.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
MODEL = DATA / "g1_dex1_model.npz"
PLANS = DATA / "plans"
ROBOT = "Unitree G1 + Dex1 (mesh fingers)"

# the G1 path's defaults, set by the runner before force_bench is imported (an explicit environment wins)
COMMON_ENV = {"G1_FINGER_COLLISION": "mesh", "G1_ARM_DAMPING": "model", "TB_TABLE": "isaac", "G1_CAMERAS": "spec"}
ROWS = {
    "5.1": {"task": "module:force_bench.appendix_tasks.turn_doorknob_open_door_flat_g1",
            "plan": "g1_door_knob_traj_s{seed}.json", "env": {"G1_FINGER_EFFORT": "80"}, "part": "door",
            "row": "turn_doorknob_open_door_flat"},
    "5.5": {"task": "module:force_bench.appendix_tasks.turn_door_handle_open_door_flat_g1",
            "plan": "g1_door_handle_traj_s{seed}.json", "env": {"G1_FINGER_EFFORT": "80"}, "part": "door",
            "row": "turn_door_handle_open_door_flat"},
    "5.2": {"task": "module:force_bench.appendix_tasks.install_light_bulb_g1", "plan": None, "env": {},
            "part": "socket", "row": "install_light_bulb"},
    "5.7": {"task": "module:force_bench.appendix_tasks.screw_nut_onto_screw_g1", "plan": None, "env": {},
            "part": "bolt", "row": "screw_nut_onto_screw"},
    "5.6": {"task": "module:force_bench.appendix_tasks.turn_dial_g1_expert", "plan": "g1_dial_iktable_s{seed}.json",
            "env": {"G1_FINGER_EFFORT": "70", "TB_DIAL_BAR_COLLIDER": "box", "G1_DIAL_REGRASP": "0",
                    "G1_DIAL_STANCE": "0.27,0.25", "G1_DIAL_TIP_LIFT": "0.0", "G1_FINGER_KD_SCALE": "1"},
            "part": "dial", "row": "turn_dial_to_target"},
}


def env_defaults(test: str) -> dict:
    """The environment this test's G1 row runs under (applied with setdefault: an explicit setting wins)."""
    return {**COMMON_ENV, **ROWS[test]["env"]}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(test: str, seed: int | None) -> dict:
    """The row, its argv and its inputs for one episode."""
    r = ROWS[test]
    plan = PLANS / r["plan"].format(seed=0 if seed is None else seed) if r["plan"] else None
    if test == "5.6" and os.environ.get("G1_DIAL_REGRASP", r["env"].get("G1_DIAL_REGRASP")) == "0":
        plan = None            # the single grasp plans online (its reach check); only the regrasp reads the IK table
    from force_bench.g1 import robot as g1robot
    inputs = {"model": {"name": MODEL.name, "sha256": _sha(MODEL)}}
    if g1robot.FINGER_COLLISION == "mesh":
        g1robot._load_hulls(str(MODEL))
        inputs["finger_hulls"] = {"name": Path(g1robot.FINGER_HULLS).name, "sha256": _sha(g1robot.FINGER_HULLS)}
    if g1robot.ARM_DAMPING == "model":
        g1robot.ARM_KD_USED.update({n: kv for n, kv in g1robot._model_kv(str(MODEL)).items()
                                    if g1robot.arm_cap(n) is not None})
        inputs["drive_gains"] = {"name": Path(g1robot.DRIVE_GAINS).name, "sha256": _sha(g1robot.DRIVE_GAINS)}
    if plan is not None:
        inputs["plan"] = {"name": plan.name, "sha256": _sha(plan)}
    return {"task": r["task"], "extra_argv": ["--model", str(MODEL)] + (["--traj", str(plan)] if plan else []),
            "inputs": inputs, "plan": json.loads(plan.read_text()) if plan else {}, "plan_path": plan}


def is_reach_miss(e: BaseException) -> bool:
    return isinstance(e, SystemExit) and ("no top-down wrist window" in str(e) or "reach miss" in str(e))


def _validate(rec) -> list:
    try:
        from sims.clockwork.record import validate
    except ImportError:
        return ["sims.clockwork.record not importable"]
    return validate(rec)


def _fork_rev():
    root = Path(__file__).resolve().parents[4]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def scene_common(rec: dict, rig=None) -> None:
    """The fields every G1 record carries (episode or reach miss)."""
    from force_bench.g1 import robot as g1robot
    st = getattr(rig, "scene_table", None) if rig is not None else None
    fcm = dict(g1robot.FINGER_COLLISION_MODEL) if g1robot.FINGER_COLLISION == "mesh" else {}
    rec["finger_collision_model"] = ({"type": "mesh", "method": fcm.get("method"),
                                      "hulls_file": Path(g1robot.FINGER_HULLS).name,
                                      "hulls_file_sha256": fcm.get("hulls_file_sha256"), "pieces": fcm.get("pieces"),
                                      "max_hull_off_mesh_mm": fcm.get("max_hull_off_mesh_mm"),
                                      "source_mesh": fcm.get("source_mesh")}
                                     if g1robot.FINGER_COLLISION == "mesh" else
                                     {"type": "box", "source": "a box per Dex1 finger link fitted to its collision mesh"})
    ve = (getattr(rig, "variant_extra", None) or {}) if rig is not None else {}
    rec["dial_bar_collider"] = ve.get("dial_bar_collider")      # 5.6: the box (type, dimensions, source); else None
    rec.update(robot=g1robot.robot_name(), robot_variant=False,
               scene_table=st if st else {"type": os.environ.get("TB_TABLE") or "contactbench",
                                          "source": "Rig.table (the Isaac table when TB_TABLE=isaac)"},
               policy_cameras={k: {kk: v for kk, v in c.items() if kk != "ros_q"} | {"ros_q": list(c["ros_q"])}
                               for k, c in g1robot.SPEC_CAMERAS.items()} if g1robot.G1_CAMERAS == "spec"
               else {"source": "the exported model's cameras (G1_CAMERAS=npz)"},
               render_version=1)


def reach_miss_record(test: str, seed, policy_id: str, ctx: dict, err: BaseException, budget, folder: Path,
                      bench: str, sim: str) -> dict:
    """A reach miss as a full contract record: no episode ran (done_reason reach_miss, zero times), the judge's one
    criterion is the reach with its numbers, and the start frame's camera videos when the row rendered them."""
    import importlib
    from force_bench.g1.robot import variant_params
    from force_bench.policy_env.adapters import FORCE_LIMIT_N, force_limit_rule
    r, t = ROWS[test], ctx["plan"]
    mod = importlib.import_module(r["task"].split(":", 1)[1])
    if t.get("reach_miss"):                       # planned offline: the planner's numbers
        numbers = {"why": t["why"], "g1_fixed_stance_m": t["fixed_stance"]}
        detail = (f"reach miss: no plan at the fixed stance {t['fixed_stance']} reaches this seed's {r['part']} "
                  f"(failed candidates by constraint: {t['why']})")
        start = {"part_shift_mm": [round(1e3 * v, 1) for v in t["door_shift_m"]], "g1_pelvis_cb_m": t["pelvis_cb"]}
    else:                                         # planned online: the row's own numbers
        numbers = {**dict(getattr(mod, "LAST_MISS", {})), "g1_fixed_stance_m": list(getattr(mod, "STANCE", ()))}
        detail, start = f"reach miss: {err}", dict(getattr(mod, "LAST_MISS", {}))
    vids = sorted(str(v.name) for v in folder.glob("camera_*.mp4"))
    rec = {"test_id": test, "seed": 0 if seed is None else int(seed), "policy_id": policy_id, "privileged": True,
           "success": False, "success_detail": detail + "; stopped: reach miss before the episode started",
           "start_seeded": seed is not None,
           "start": {**start, "g1_fixed_stance_m": numbers["g1_fixed_stance_m"],
                     "arm": t.get("arm") or getattr(mod, "SIDE", None),
                     "note": f"the G1 is fixed relative to the table; the seed moves only the {r['part']}"},
           "simulated": False, "inputs": ctx["inputs"], "robot_variant_params": variant_params(),
           "benchmark": bench, "task": r["row"], "sim": sim, "mode": "lockstep", "fork_rev": _fork_rev(),
           "budget_s": budget, "attempts": 1, "regrasps": 0, "retry_events": [], "irreversible": [],
           "done_reason": "reach_miss", "t_wall_s": 0.0, "t_world_s": 0.0, "t_host_s": 0.0, "policy_latency_s": 0.0,
           "progress": 0.0, "force_limit_N": FORCE_LIMIT_N, "force_limit_rule": force_limit_rule(),
           "judge": {"pass": False, "criteria": {"reach": {"pass": False, **numbers}}, "finite": True},
           "video": vids, "start_frame_video": bool(vids),
           "physics_qa": {"max_single_frame_contact_N": None, "flags": [],
                          "max_single_frame_contact_detail": {"full": None, "normal": None,
                                                              "measure": "a reach miss: no contact was simulated"}},
           "record_note": "a reach miss: no grasp reaches this seed's part at the fixed stance, so no episode ran; "
                          "the videos hold the seeded start scene's first frame"}
    scene_common(rec)
    return rec


def max_single_frame_contact(cp) -> dict:
    def one(p, other_key):
        if not p:
            return None
        return {"force_N": p["force_N"], other_key: p.get("normal_N" if other_key == "normal_N" else "full_N"),
                "t": p["t"], "pair": [p["shape_a"], p["shape_b"]], "bodies": [p.get("body_a"), p.get("body_b")],
                "phase": p.get("phase")}
    if cp is None:
        return {"full": None, "normal": None, "measure": "no contact monitor on this row"}
    return {"full": one(cp.peak, "normal_N"), "normal": one(getattr(cp, "peak_normal", None), "full_N"),
            "measure": "per frame, the robot's contacts summed per shape pair; the episode's strongest pair. full: "
                       "sum of |f| (the > 500 N flag's number); normal: sum of |f.n|",
            "flag_threshold_N": 500.0, "flag_basis": "full"}


def part_ejected(folder, limit_m: float = 1.0) -> list:
    import numpy as np
    out = []
    for f in sorted(Path(folder).glob("**/rollout_states.npz")):
        d = np.load(f, allow_pickle=True)
        lab = [str(x) for x in d["body_labels"]]
        q = d["body_q"][:, :, :3]
        disp = np.linalg.norm(q - q[0][None], axis=2)
        for b, n in enumerate(lab):
            if n.startswith("g1/"):
                continue
            k = int(disp[:, b].argmax())
            if disp[k, b] > limit_m or disp[-1, b] > limit_m:
                out.append({"body": n, "max_m": round(float(disp[k, b]), 3), "t_max": round(float(d["t"][k]), 3),
                            "final_m": round(float(disp[-1, b]), 3)})
    return out


def extras(rec: dict, res: dict, folder: Path, test: str, seed, ctx: dict) -> dict:
    """The G1 fields on a simulated episode's record (in place): placement, settings with sources, caps read back,
    rule A per finger (normal and full), the single-frame contact, torques and the contract's physics flags."""
    from force_bench.g1 import robot as g1robot
    from force_bench.g1.robot import variant_params
    rig = res["episode"].rig
    t = {**ctx["plan"], **dict(getattr(rig, "g1_plan", {}))}
    if os.environ.get("G1_PAD_MU") not in (None, "1.0", "1") and rec.get("published"):
        raise SystemExit("G1_PAD_MU is a diagnostic override: published episodes use the model's pad friction")
    scene_common(rec, rig)
    rec.update(privileged=True, robot_variant_params=variant_params(),
               placement={k: t.get(k) for k in ("pelvis_cb", "arm", "waist_pitch", "gamma_deg", "orient",
                                                "door_shift_m", "stance_step_m")})
    try:
        rec["solver_options_runtime"] = g1robot.solver_options_readback(rig.solver)
    except Exception as e:  # noqa: BLE001
        rec["solver_options_runtime"] = {"error": repr(e)}
    argv_ = [Path(v).name if v in (str(MODEL), str(ctx.get("plan_path"))) else v for v in (rec.get("row_argv") or [])]
    rec["row_argv"], rec["inputs"] = argv_, ctx["inputs"]
    sh = t.get("door_shift_m") or [0.0, 0.0]
    rec["start"] = {"door_shift_mm": [round(1e3 * v, 1) for v in sh], "g1_pelvis_cb_m": t.get("pelvis_cb"),
                    "g1_fixed_stance_m": t.get("fixed_stance"), "arm": t.get("arm"),
                    "note": "the G1 is fixed relative to the table" + (
                        f"; the seed moves only the {ROWS[test]['part']}" if seed is not None else
                        "; this row has no seeded start: every replicate starts the same")}
    strokes = getattr(rig, "dial_strokes", None)
    if strokes:
        rec["strokes"] = strokes
        rec["regrasps"] = max(0, sum(1 for s_ in strokes if "grasp_yaw_deg" in s_) - 1)
        rec["strokes_note"] = "regrasps counts the planned re-closings (strokes - 1); attempts stays 1 (D1)"
        slips = [s_ for s_ in strokes if s_.get("drift_while_open_deg") is not None
                 and abs(s_["drift_while_open_deg"]) > 7.5]
        rec["brake_held_between_strokes"] = not slips
        if slips and rec["success"]:
            rec["success"] = False
            rec["success_detail"] = (f"failed: the dial drifted {slips[0]['drift_while_open_deg']} deg while the jaws "
                                     "were open between strokes (> half a tick, 7.5 deg); " + rec["success_detail"])
    ve = getattr(rig, "variant_extra", None)
    if ve:
        rec["robot_variant_params"].update(ve)
    fp = getattr(rig, "finger_params", None)
    if fp is not None:
        rec["robot_variant_params"]["finger_actuators_sim_readback"] = fp
    if getattr(rig, "arm_caps_sim", None) is not None:
        rec["robot_variant_params"]["arm_effort_caps_sim_readback"] = rig.arm_caps_sim
    if getattr(rig, "arm_caps", None) is not None:
        rec["robot_variant_params"]["arm_effort_caps_Nm"] = rig.arm_caps
        rec["robot_variant_params"]["arm_effort_cap_headroom"] = getattr(rig, "effort_budget", None)
    qa = rec.setdefault("physics_qa", {})
    jqa = (res.get("judge") or {}).get("physics_qa")
    if jqa:
        qa.update(jqa)
    tm = getattr(rig, "torque_monitor", None)
    if tm is not None:
        pk = tm.summary()
        rec["peak_joint_torque_Nm"] = pk
        qa["max_joint_torque_frac"] = {n: round(v["peak_Nm"] / v["limit_Nm"], 4) for n, v in pk.items()}
        qa["actuator_at_cap_frames"] = {n: v["frames_actuator_at_cap"] for n, v in pk.items()}
        qa["joint_torque_note"] = ("per arm joint, the whole episode: |PD actuator, clipped at its cap, + gravity "
                                   "feed-forward| over the G1's URDF limit")
    lim = float(rec.get("force_limit_N") or 150.0)

    def per_finger(labels, key="force"):
        per = {}
        for lb in labels:
            hist = rig.probes[lb].history
            if not hist:
                continue
            fs = [h_.get(key, h_["force"]) for h_ in hist]
            k = max(range(len(fs)), key=lambda i: fs[i])
            run = best = 0
            for v in fs:
                run = run + 1 if v > lim else 0
                best = max(best, run)
            per[lb] = {"force_N": round(fs[k], 1), "t": round(float(hist[k]["t"]), 3),
                       "frames_over_limit": int(sum(v > lim for v in fs)), "longest_run": best}
        if not per:
            return None
        top = max(per, key=lambda k_: per[k_]["force_N"])
        return {"force_N": per[top]["force_N"], "finger": top, "t": per[top]["t"],
                "frames_over_limit": sum(v["frames_over_limit"] for v in per.values()),
                "longest_run": max(v["longest_run"] for v in per.values()), "per_finger": per}
    rule_a = getattr(rig, "rule_a_probes", None)
    if rule_a:
        pk_, full_ = per_finger(rule_a), per_finger(rule_a, "force_full")
        if pk_:
            for k_, v_ in pk_["per_finger"].items():
                v_["full_magnitude_peak_N"] = full_["per_finger"][k_]["force_N"]
                v_["full_frames_over_limit"] = full_["per_finger"][k_]["frames_over_limit"]
                v_["full_longest_run"] = full_["per_finger"][k_]["longest_run"]
            rec["peak_pad_force_N"] = {**pk_, "full_magnitude_peak_N": full_["force_N"],
                                       "measure": "per finger (hand_Link{k}_2 + _3) against every non-robot shape: "
                                                  "the sum of contact NORMAL-force magnitudes |f.n| (rule A); the full "
                                                  "|f| sum alongside (normal <= full)"}
        rec["force_rule"] = rec["force_limit_rule"] = "normal-magnitude-sum (contract #1773, 19 Nov)"
        rec["force_rule_verified"], rec["force_rule_basis"] = True, "measured"
        cols = [(c, "force", c.replace("/", "_") + "_normal_N") for c in rule_a] + \
               [(c, "force_full", c.replace("/", "_") + "_magnitude_N") for c in rule_a]
        hs = [rig.probes[c].history for c, _, _ in cols]
        n_ = min(len(h_) for h_ in hs) if hs else 0
        with open(folder / "finger_forces.csv", "w") as fh:
            fh.write("t," + ",".join(nm for _, _, nm in cols) + "\n")
            for i in range(n_):
                fh.write(f"{hs[0][i]['t']:.4f}," + ",".join(f"{h_[i].get(k_, h_[i]['force']):.3f}"
                                                            for h_, (_, k_, _) in zip(hs, cols)) + "\n")
        rec["finger_forces_csv"] = "finger_forces.csv"
    flags = []
    cp = getattr(rig, "contact_peak", None)
    msf = max_single_frame_contact(cp)
    qa["max_single_frame_contact_N"] = (msf["full"]["force_N"] if msf["full"] else 0.0) if cp is not None else None
    qa["max_single_frame_contact_detail"] = msf
    if msf["full"] and msf["full"]["force_N"] > 500.0:
        flags.append({"flag": "single_frame_contact_over_500N", "force_N": msf["full"]["force_N"],
                      "t": msf["full"]["t"], "pair": msf["full"]["pair"], "phase": msf["full"]["phase"]})
    if tm is not None:
        for n_, v_ in tm.summary().items():
            if v_["frames_actuator_at_cap"] > 10:
                flags.append({"flag": "actuator_at_cap_over_10_frames", "joint": n_,
                              "frames": v_["frames_actuator_at_cap"]})
    ej = part_ejected(folder)
    if ej:
        flags.append({"flag": "part_ejected", "bodies": ej})
    qa["flags"] = sorted(set(qa.get("flags") or []) | {f_["flag"] for f_ in flags})
    if flags:
        qa["flag_details"] = list(qa.get("flag_details") or []) + flags
    hit = sorted({"single_frame_contact_over_500N", "actuator_at_cap_over_10_frames", "part_ejected"}
                 & set(qa["flags"]))
    if rec["success"] and hit:
        rec["flagged_pass"] = True
        rec["success_detail"] = (f"FLAGGED: the judge's criteria pass, but physics_qa flags {hit} stand (a flagged pass "
                                 "does not count); " + rec["success_detail"])
    if cp is not None:
        rec["peak_contact"] = cp.summary()
    return rec
