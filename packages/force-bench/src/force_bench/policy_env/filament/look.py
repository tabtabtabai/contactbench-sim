"""Render version 4's materials for the kit and props: physically based parameters on the model's colours (the kit
look, display sRGB), and crease-split normals (round parts shade smoothly, sharp edges stay sharp).  The robot, room and
table materials come from ``isaac_scene``.
"""
from __future__ import annotations

import re

import numpy as np

PBR = {  # class: (roughness, metallic, specular)
    "pla": (0.42, 0.0, 0.5),          # matte-satin printed filament
    "mat": (0.80, 0.0, 0.35),         # the white ribbed board
    "desk": (0.55, 0.0, 0.45),        # maple desk
    "wall": (0.92, 0.0, 0.25),
    "robot_paint": (0.35, 0.0, 0.5),
    "robot_dark": (0.45, 0.2, 0.5),
}
KIT_RULES = [
    (r"^render_v2/floor$", "desk"), (r"^table$", "mat"), (r"^render_v2/wall", "wall"),
    (r"^kit/", "pla"), (r"^bolt$", "pla"), (r"^nut_shape$", "pla"),
]
CREASE_DEG = 35.0


def material_class(label: str, body: str, color) -> str:
    for pat, cls in KIT_RULES:
        if re.search(pat, label):
            return cls
    if body.startswith(("fr3/", "g1/")) or label.startswith("shape_"):
        return "robot_paint" if float(np.mean(color)) > 0.6 else "robot_dark"
    return "pla"


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def crease_normals(V, F, crease_deg=CREASE_DEG):
    """Face-varying normals: each corner averages the area-weighted normals of the faces at its vertex that lie
    within crease_deg of its own face."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fu = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    cosc = np.cos(np.radians(crease_deg))
    corners = F.reshape(-1)
    face_of = np.repeat(np.arange(len(F)), 3)
    order = np.argsort(corners, kind="stable")
    sv, sf = corners[order], face_of[order]
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    ends = np.r_[starts[1:], len(sv)]
    out = np.zeros((len(corners), 3))
    for s, e in zip(starts, ends):
        faces = sf[s:e]
        m = (fu[faces] @ fu[faces].T) >= cosc
        acc = m.astype(float) @ fn[faces]
        out[order[s:e]] = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-20)
    return out.astype(np.float32)



def box_mesh(h):
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return c, F
