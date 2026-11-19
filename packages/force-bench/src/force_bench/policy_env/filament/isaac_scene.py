"""The Unitree competition Isaac scene for render version 4: room, table look, lights and the Isaac G1's materials,
placed around a ContactBench G1 by its pelvis.  Look only: no body is moved and no collider is added.

Source: matrix unibot-scene-spec.md (sections 1-4) from the construct image c74074b8: ``bright_room.usd`` and
``plain_table.usd`` (``sims/isaac/make_props.py:529-616``, read back with pxr) and the robot USD's OmniPBR materials.
Colours are linear (UsdPreviewSurface / MDL).  The table is drawn where the Isaac scene puts it relative to the pelvis,
which is where ``TB_TABLE=isaac`` (rig.py) puts ContactBench's physics table, so look and physics agree.

The exposure, the key/dome balance and two robot-material scales are FITTED to real G1 head-camera frames (region
means of table, wall, silver arm and gripper; error 57 -> 20 in 8-bit units against the Isaac-RTX-like first fit);
they are the only values here not read from the Isaac assets.
"""
from __future__ import annotations

import numpy as np

ISAAC_PELVIS = np.array([-0.15, 0.0, 0.76])          # robot_configs.py:252-253 (construct c74074b8), facing +y
ISAAC_PELVIS_YAW = np.pi / 2

# bright_room.usd (Isaac world): name, centre, full size, diffuse (linear), roughness, metallic, emissive
_ROOM = [
    ("floor", (0, 0, -0.01), (24, 24, 0.02), (0.44, 0.44, 0.44), 0.75, 0.0, None),
    ("wall_back_left", (-5.1, 1.15, 1.75), (5.8, 0.2, 3.5), (0.71, 0.72, 0.74), 0.9, 0.0, None),
    ("wall_back_right", (4.4, 1.15, 1.75), (7.2, 0.2, 3.5), (0.71, 0.72, 0.74), 0.9, 0.0, None),
    ("wall_back_below", (-0.7, 1.15, 0.39), (3.0, 0.2, 0.78), (0.71, 0.72, 0.74), 0.9, 0.0, None),
    ("wall_back_above", (-0.7, 1.15, 3.05), (3.0, 0.2, 0.9), (0.71, 0.72, 0.74), 0.9, 0.0, None),
    ("frame_l", (-2.2, 1.15, 1.69), (0.06, 0.24, 1.88), (0.84, 0.84, 0.83), 0.6, 0.0, None),
    ("frame_r", (0.8, 1.15, 1.69), (0.06, 0.24, 1.88), (0.84, 0.84, 0.83), 0.6, 0.0, None),
    ("frame_t", (-0.7, 1.15, 2.6), (3.06, 0.24, 0.06), (0.84, 0.84, 0.83), 0.6, 0.0, None),
    ("sill", (-0.7, 1.12, 0.78), (3.1, 0.3, 0.05), (0.84, 0.84, 0.83), 0.6, 0.0, None),
    ("mullion", (-0.7, 1.15, 1.69), (0.04, 0.22, 1.82), (0.84, 0.84, 0.83), 0.6, 0.0, None),
    ("sky", (-0.7, 1.75, 1.7), (3.9, 0.05, 2.6), (0.9, 0.95, 0.88), 1.0, 0.0, (1.5, 1.58, 1.4)),
    ("tree_0", (0.55, 1.62, 1.15), (0.75, 0.12, 0.85), (0.55, 0.66, 0.48), 1.0, 0.0, None),
    ("tree_1", (0.85, 1.55, 1.8), (0.55, 0.12, 1.3), (0.64, 0.74, 0.54), 1.0, 0.0, None),
    ("tree_1b", (0.45, 1.68, 2.0), (0.4, 0.1, 0.7), (0.55, 0.66, 0.48), 1.0, 0.0, None),
    ("tree_2", (-2.15, 1.62, 1.1), (0.55, 0.12, 0.8), (0.55, 0.66, 0.48), 1.0, 0.0, None),
    ("tree_3", (-1.95, 1.55, 2.0), (0.5, 0.12, 0.95), (0.64, 0.74, 0.54), 1.0, 0.0, None),
    ("tree_3b", (-2.3, 1.68, 1.7), (0.35, 0.1, 0.6), (0.55, 0.66, 0.48), 1.0, 0.0, None),
]
# plain_table.usd (asset frame): top slab and legs
_TABLE = [
    ("top", (0, 0, 0.715), (1.7, 0.85, 0.05), (0.72, 0.69, 0.61), 0.55, 0.0, None),
    *[(f"leg_{i}", (sx * 0.78, sy * 0.35, 0.345), (0.06, 0.06, 0.69), (0.55, 0.55, 0.57), 0.4, 0.6, None)
      for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)])],
]
# the table's placement in each task (spec section 9); ColorArrangement is the baseline
TABLES = {"color_arrangement": {"pos": (-0.10, 0.62, 0.0973), "scale": (0.85, 1.0, 1.0)},
          "fold_towel": {"pos": (-0.10, 0.47, 0.0973), "scale": (0.85, 0.7765, 1.0)}}
DOME = {"color": (0.74, 0.76, 0.72)}                  # color_arrangement_env_cfg.py:100-103, no texture
KEY = {"color": (1.0, 0.97, 0.90), "angle_deg": 8.0, "travel_isaac": (0.0, -0.343, -0.939)}   # :104-111
HIDE = ("ground_plane", "table", "table_base")        # ContactBench stand-ins drawn instead by the Isaac room/table
HIDE_PREFIX = ("render_v2/",)

# the look fitted to real G1 head frames (see the module docstring)
LOOK = {"ibl": 25000.0, "sun_lux": 35000.0, "ev100": 14.8, "tonemap": "aces", "ssr": 1,
        "body_metal_scale": 1.6, "dex1_albedo": 0.015}

# the Isaac G1 USD's OmniPBR materials (albedo = diffuse_color_constant x diffuse_tint), (albedo, metallic, roughness)
_OMNI = {
    "DefaultMaterial": ((0.56, 0.56, 0.5396), 1.0, 0.6),
    "DefaultMaterial_0": ((0.0154, 0.0148, 0.0148), 1.0, 0.5),
    "material_CAD1EE": ((0.0149, 0.0150, 0.0154), 1.0, 0.5),
    "material_E5EAED": ((0.0299, 0.0305, 0.0309), 1.0, 0.5),
    "material_4C4C4C": ((0.9653, 0.9503, 0.9503), 0.0, 0.5),
    "OmniPBR": ((0.6293, 0.5978, 0.5978), 0.0, 0.5),
}
_DARK0 = ("head_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_hip_pitch_link", "right_hip_pitch_link",
          "waist_yaw_link")


def robot_material(body_label: str):
    """(linear albedo, metallic, roughness) for a G1 body, from the USD's per-link bindings, with the fitted scales."""
    link = body_label.split("/", 1)[-1]
    if link in _DARK0:
        name = "DefaultMaterial_0"
    elif link == "logo_link":
        name = "OmniPBR"
    elif "hand" in link and link.endswith("_3"):
        name = "material_4C4C4C"
    elif "hand" in link and link.endswith("_2"):
        name = "material_E5EAED"
    elif "hand" in link:
        name = "material_CAD1EE"
    else:
        name = "DefaultMaterial"
    alb, met, rgh = _OMNI[name]
    if name == "DefaultMaterial":
        alb = tuple(min(0.95, v * LOOK["body_metal_scale"]) for v in alb)
    elif name in ("material_CAD1EE", "material_E5EAED"):
        alb = (LOOK["dex1_albedo"],) * 3
    return alb, met, rgh


def hidden(label: str) -> bool:
    return label in HIDE or label.startswith(HIDE_PREFIX)


def _quat_mat(q):  # xyzw
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def scene(pelvis_q, task: str = "color_arrangement") -> dict:
    """Boxes in ContactBench world (centre, half extents, 3x3 rotation, material) and the key light's travel."""
    q = np.asarray(pelvis_q, float)
    R = _quat_mat(q[3:]) @ _rz(-ISAAC_PELVIS_YAW)
    t = q[:3] - R @ ISAAC_PELVIS
    tab = TABLES[task]
    boxes = []
    for group, items in (("isaac_room", _ROOM), ("isaac_table", _TABLE)):
        for name, c, size, alb, rgh, met, emi in items:
            c, size = np.asarray(c, float), np.asarray(size, float)
            if group == "isaac_table":
                c = np.asarray(tab["scale"]) * c + np.asarray(tab["pos"])
                size = np.asarray(tab["scale"]) * size
            boxes.append({"name": f"{group}/{name}", "center": R @ c + t, "half": size / 2, "R": R,
                          "albedo": alb, "roughness": rgh, "metallic": met, "emissive": emi})
    travel = R @ np.asarray(KEY["travel_isaac"])
    return {"boxes": boxes, "key_travel": travel / np.linalg.norm(travel)}
