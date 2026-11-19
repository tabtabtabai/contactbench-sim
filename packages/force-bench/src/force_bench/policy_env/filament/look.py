"""Render version 4's look: physically based parameters on top of render v3's colours and textures.

Base colours come from the model as the recorder leaves them (v3's kit colours, display sRGB); this adds what the
Newton raytracer can't draw: roughness, metalness and specular per material class, normal maps derived from v3's
procedural textures, and crease-split normals (round parts shade smoothly, sharp edges stay sharp).
"""
from __future__ import annotations

import io
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
# v3 textures (render_v2._texture) on box-projected UVs, at v3's tiling in cycles per metre
TEX_SHAPES = {r"^table$": ("mat", 3.3), r"^render_v2/floor$": ("desk", 1.5)}
PLA_TEX = ("pla", 40.0)
NORMAL_STRENGTH = {"mat": 0.7, "desk": 0.6, "pla": 0.4}
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


def box_project_st(P, N, cycles):
    """Face-varying UVs: each corner takes the two coordinates orthogonal to its normal's dominant axis."""
    keep = np.array([[1, 2], [0, 2], [0, 1]])[np.argmax(np.abs(N), axis=1)]
    return (np.take_along_axis(P, keep, axis=1) * cycles).astype(np.float32)


def box_mesh(h):
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return c, F


def _png(arr) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def textures() -> dict:
    """{kind: {"albedo": png, "normal": png[, "mr": png]}} from render v3's own procedural textures."""
    from force_bench import render_v2
    out = {}
    for kind, k in NORMAL_STRENGTH.items():
        t = np.asarray(render_v2._texture(kind, seed=0))[..., :3].astype(np.uint8)
        a = t.mean(axis=2).astype(np.float32) / 255.0
        h = (a - a.mean()) / max(float(a.std()), 1e-6)
        dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / 2
        dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / 2
        s = k / 255.0 * 40
        n = np.dstack([-dx * s, -dy * s, np.ones_like(h)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        out[kind] = {"albedo": _png(t), "normal": _png(((n * 0.5 + 0.5) * 255).astype(np.uint8))}
        if kind == "pla":   # roughness 0.30-0.60 from the texture's faint mottling (glTF: G = roughness, B = metal)
            rough = np.clip(PBR["pla"][0] + (a - 0.998) * 10.0, 0.30, 0.60)
            mr = np.zeros(a.shape + (3,), np.uint8)
            mr[..., 1] = (rough * 255).astype(np.uint8)
            out[kind]["mr"] = _png(mr)
    return out
