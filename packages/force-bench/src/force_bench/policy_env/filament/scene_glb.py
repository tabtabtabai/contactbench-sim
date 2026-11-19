"""Write a running ContactBench rig's scene as a binary glTF (.glb) for the Filament server.

One node per body (named ``b<index>_<label>``, the order of ``model.body_label``) holding its visible shapes in
body-local coordinates, so the server animates the scene by setting body transforms only.  World shapes are root
nodes.  Colours are the model's (the kit look and render v3 colours applied by the caller first), turned into PBR
materials by ``look``.  No physics is read or changed.
"""
from __future__ import annotations

import json
import re
import struct

import numpy as np

from force_bench.policy_env.filament import look


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def body_node_names(model) -> list[str]:
    return [f"b{b}_" + re.sub(r"[^A-Za-z0-9_]", "_", str(lbl or "")) for b, lbl in enumerate(model.body_label)]


def write_scene_glb(model, body_q, out_path: str, textured: bool = True) -> list[str]:
    """Write ``out_path``; returns the body node names in body order (the server's names file)."""
    import newton
    visible = int(newton.ShapeFlags.VISIBLE)
    geo = {int(getattr(newton.GeoType, k)): k for k in dir(newton.GeoType)
           if not k.startswith("_") and isinstance(getattr(newton.GeoType, k), int)}
    labels = [str(x or "") for x in model.shape_label]
    blabels = [str(x or "") for x in model.body_label]
    sbody = model.shape_body.numpy()
    stype = model.shape_type.numpy()
    sscale = model.shape_scale.numpy()
    stf = model.shape_transform.numpy()
    sflags = model.shape_flags.numpy()
    scolor = model.shape_color.numpy()
    src = list(model.shape_source)
    names = body_node_names(model)
    tex = look.textures() if textured else {}

    buf = bytearray()
    views, accessors, meshes, materials, nodes, images, textures = [], [], [], [], [], [], []
    mat_index, tex_index = {}, {}

    def view(b: bytes, target=None):
        nonlocal buf
        while len(buf) % 4:
            buf += b"\0"
        views.append({"buffer": 0, "byteOffset": len(buf), "byteLength": len(b), **({"target": target} if target else {})})
        buf += b
        return len(views) - 1

    def accessor(arr, typ):
        arr = np.ascontiguousarray(arr, np.float32)
        acc = {"bufferView": view(arr.tobytes(), 34962), "componentType": 5126, "count": int(arr.shape[0]), "type": typ}
        if typ == "VEC3":
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    def image(key, png):
        if key not in tex_index:
            images.append({"bufferView": view(png), "mimeType": "image/png"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            tex_index[key] = len(textures) - 1
        return tex_index[key]

    def material(color, cls, kind):
        key = (cls, kind, tuple(np.round(color, 3)))
        if key not in mat_index:
            rough, metal, _ = look.PBR[cls]
            m = {"name": f"m{len(materials)}_{cls}", "pbrMetallicRoughness": {
                "baseColorFactor": [*map(float, look.srgb_to_linear(color)), 1.0],
                "metallicFactor": metal, "roughnessFactor": rough}}
            if kind is not None:
                m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": image((kind, "a"), tex[kind]["albedo"])}
                m["normalTexture"] = {"index": image((kind, "n"), tex[kind]["normal"])}
                if "mr" in tex[kind]:
                    m["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": image((kind, "mr"), tex[kind]["mr"])}
                    m["pbrMetallicRoughness"]["roughnessFactor"] = 1.0
            materials.append(m)
            mat_index[key] = len(materials) - 1
        return mat_index[key]

    for b, nm in enumerate(names):
        q = np.asarray(body_q[b], float)
        nodes.append({"name": nm, "children": [], "translation": q[:3].tolist(), "rotation": q[3:].tolist()})
    roots = list(range(len(names)))
    for i in range(len(labels)):
        if not int(sflags[i]) & visible:
            continue
        typ, sc = geo.get(int(stype[i]), ""), np.asarray(sscale[i], float)
        if typ == "MESH" and src[i] is not None and hasattr(src[i], "vertices"):
            V = np.asarray(src[i].vertices, float) * sc
            F = np.asarray(src[i].indices, np.int64).reshape(-1, 3)
        elif typ == "BOX":
            V, F = look.box_mesh(sc[:3])
        else:
            continue
        N = look.crease_normals(V, F)
        P = V[F.reshape(-1)]
        tf = np.asarray(stf[i], float)
        Rl, tl = _quat_mat(tf[3:]), tf[:3]
        b = int(sbody[i])
        Pl, Nl = P @ Rl.T + tl, N @ Rl.T                       # in the body's frame (or the world, for b < 0)
        color = np.asarray(scolor[i], float)
        body = blabels[b] if b >= 0 else ""
        cls = look.material_class(labels[i], body, color)
        kind, uv = None, None
        if textured:
            spec = next((v for pat, v in look.TEX_SHAPES.items() if re.search(pat, labels[i])), None)
            if spec is not None and typ == "BOX" and b < 0:
                kind, uv = spec[0], look.box_project_st(Pl, Nl, spec[1])
            elif cls == "pla" and typ == "MESH":
                kind, uv = look.PLA_TEX[0], look.box_project_st(P, N, look.PLA_TEX[1])
        attrs = {"POSITION": accessor(Pl, "VEC3"), "NORMAL": accessor(Nl, "VEC3")}
        if uv is not None:
            attrs["TEXCOORD_0"] = accessor(uv, "VEC2")
        meshes.append({"name": re.sub(r"[^A-Za-z0-9_]", "_", labels[i]) or f"s{i}",
                       "primitives": [{"attributes": attrs, "material": material(color, cls, kind), "mode": 4}]})
        nodes.append({"mesh": len(meshes) - 1, "name": f"s{i}"})
        (nodes[b]["children"] if b >= 0 else roots).append(len(nodes) - 1)

    gltf = {"asset": {"version": "2.0", "generator": "contactbench render v4 (filament)"}, "scene": 0,
            "scenes": [{"nodes": roots}], "nodes": nodes, "meshes": meshes, "materials": materials,
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": 0}]}
    if images:
        gltf.update(images=images, textures=textures,
                    samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
    while len(buf) % 4:
        buf += b"\0"
    gltf["buffers"][0]["byteLength"] = len(buf)
    js = json.dumps(gltf).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(buf), 0x004E4942)); f.write(bytes(buf))
    return names
