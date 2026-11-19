"""Write a running ContactBench G1 rig's scene as a binary glTF (.glb) for the Filament server (render version 4).

One node per body (named ``b<index>_<label>``, the order of ``model.body_label``) holding its visible shapes in
body-local coordinates, so the server animates the scene by setting body transforms only.  World shapes are root nodes.
The Isaac challenge scene is added around the robot (``isaac_scene``): its room and table boxes, and the Isaac G1's
materials on the robot; ContactBench's own ground and table stand-ins are not drawn.  Kit parts keep the model's
colours (the kit look, applied by the caller first) as PBR materials (``look``).  No physics is read or changed.
"""
from __future__ import annotations

import json
import re
import struct

import numpy as np

from force_bench.policy_env.filament import isaac_scene, look


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def body_node_names(model) -> list[str]:
    return [f"b{b}_" + re.sub(r"[^A-Za-z0-9_]", "_", str(lbl or "")) for b, lbl in enumerate(model.body_label)]


def write_scene_glb(model, body_q, out_path: str, task: str = "color_arrangement") -> dict:
    """Write ``out_path``; returns {"names": body node names in body order, "key_travel": the key light's direction}."""
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
    sc = isaac_scene.scene(np.asarray(body_q[blabels.index("g1/pelvis")], float), task)

    buf = bytearray()
    views, accessors, meshes, materials, nodes = [], [], [], [], []
    mat_index = {}

    def view(b: bytes, target=None):
        nonlocal buf
        while len(buf) % 4:
            buf += b"\0"
        views.append({"buffer": 0, "byteOffset": len(buf), "byteLength": len(b), **({"target": target} if target else {})})
        buf += b
        return len(views) - 1

    def accessor(arr):
        arr = np.ascontiguousarray(arr, np.float32)
        accessors.append({"bufferView": view(arr.tobytes(), 34962), "componentType": 5126, "count": int(arr.shape[0]),
                          "type": "VEC3", "min": arr.min(0).tolist(), "max": arr.max(0).tolist()})
        return len(accessors) - 1

    def material(albedo_linear, metal, rough, emissive=None):
        key = (tuple(np.round(albedo_linear, 4)), round(float(metal), 3), round(float(rough), 3), tuple(emissive or ()))
        if key not in mat_index:
            m = {"name": f"m{len(materials)}", "pbrMetallicRoughness": {
                "baseColorFactor": [*map(float, albedo_linear), 1.0], "metallicFactor": float(metal),
                "roughnessFactor": float(rough)}}
            if emissive:
                e = np.asarray(emissive, float)
                m["emissiveFactor"] = (e / e.max()).tolist()
                m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": float(e.max())}}
            materials.append(m)
            mat_index[key] = len(materials) - 1
        return mat_index[key]

    def add_mesh(name, P, N, mi, parent):
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": accessor(P), "NORMAL": accessor(N)},
                                                     "material": mi, "mode": 4}]})
        nodes.append({"mesh": len(meshes) - 1, "name": name})
        (nodes[parent]["children"] if parent is not None else roots).append(len(nodes) - 1)

    for b, nm in enumerate(names):
        q = np.asarray(body_q[b], float)
        nodes.append({"name": nm, "children": [], "translation": q[:3].tolist(), "rotation": q[3:].tolist()})
    roots = list(range(len(names)))
    for i in range(len(labels)):
        if not int(sflags[i]) & visible or isaac_scene.hidden(labels[i]):
            continue
        typ, s = geo.get(int(stype[i]), ""), np.asarray(sscale[i], float)
        if typ == "MESH" and src[i] is not None and hasattr(src[i], "vertices"):
            V = np.asarray(src[i].vertices, float) * s
            F = np.asarray(src[i].indices, np.int64).reshape(-1, 3)
        elif typ == "BOX":
            V, F = look.box_mesh(s[:3])
        else:
            continue
        N = look.crease_normals(V, F)
        P = V[F.reshape(-1)]
        tf = np.asarray(stf[i], float)
        Rl, tl = _quat_mat(tf[3:]), tf[:3]
        b = int(sbody[i])
        body = blabels[b] if b >= 0 else ""
        color = np.asarray(scolor[i], float)
        if body.startswith("g1/"):
            mi = material(*isaac_scene.robot_material(body))
        else:
            rough, metal, _ = look.PBR[look.material_class(labels[i], body, color)]
            mi = material(look.srgb_to_linear(color), metal, rough)
        add_mesh(re.sub(r"[^A-Za-z0-9_]", "_", labels[i]) or f"s{i}", P @ Rl.T + tl, N @ Rl.T, mi, b if b >= 0 else None)
    for bx in sc["boxes"]:
        V, F = look.box_mesh(np.asarray(bx["half"], float))
        N = look.crease_normals(V, F)
        R = np.asarray(bx["R"], float)
        add_mesh(bx["name"].replace("/", "_"), V[F.reshape(-1)] @ R.T + bx["center"], N @ R.T,
                 material(bx["albedo"], bx["metallic"], bx["roughness"], bx["emissive"]), None)

    gltf = {"asset": {"version": "2.0", "generator": "contactbench render v4 (filament, Isaac scene)"}, "scene": 0,
            "scenes": [{"nodes": roots}], "nodes": nodes, "meshes": meshes, "materials": materials,
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": 0}],
            "extensionsUsed": ["KHR_materials_emissive_strength"]}
    while len(buf) % 4:
        buf += b"\0"
    gltf["buffers"][0]["byteLength"] = len(buf)
    js = json.dumps(gltf).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(buf), 0x004E4942)); f.write(bytes(buf))
    return {"names": names, "key_travel": sc["key_travel"].tolist()}
