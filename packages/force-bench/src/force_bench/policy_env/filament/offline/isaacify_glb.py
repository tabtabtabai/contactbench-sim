"""Put a G1 glb (body nodes b<idx>_g1_<link>, optional "deform/towel" and other static nodes) into the Isaac challenge
scene, using the contactbench-sim fork's render-v4 scene module (force_bench.policy_env.filament.isaac_scene): append
the room and table boxes for TASK placed by the g1/pelvis pose (frame 0 of STATES), give robot meshes the Isaac G1
materials with the real-frame fitted scales, drop lights and any node named "table".  Look only.
Writes OUT.glb, OUT.scene.json (key light direction + a third-person camera) and OUT.bodies.txt.

usage: PYTHONPATH=<fork>/packages/force-bench/src python isaacify_glb.py IN.glb STATES.npz TASK OUT.glb
       TASK: fold_towel | color_arrangement
"""
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

from force_bench.policy_env.filament import isaac_scene
from force_bench.policy_env.filament.look import box_mesh, crease_normals

# third-person view (NOT from the scene spec: no scored Dex1 task has one): eye/target in the pelvis frame
THIRD = {"eye_pelvis": (0.82, 1.15, 0.60), "target_pelvis": (0.30, -0.12, 0.02), "focal_mm": 16.5, "aperture_mm": 20.0}


def _quat_xyzw(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]


def main():
    src, states, task, out = sys.argv[1:5]
    raw = Path(src).read_bytes()
    jl = struct.unpack("<I", raw[12:16])[0]
    g = json.loads(raw[20:20 + jl])
    bl = struct.unpack("<I", raw[20 + jl:24 + jl])[0]
    buf = bytearray(raw[28 + jl:28 + jl + bl])
    R = np.load(states, allow_pickle=True)
    labels = [str(x) for x in R["body_labels"]]
    pel = np.asarray(R["body_q"][0, labels.index("g1/pelvis")], float)
    sc = isaac_scene.scene(pel, task)

    def acc(arr):
        nonlocal buf
        arr = np.ascontiguousarray(arr, np.float32)
        while len(buf) % 4:
            buf += b"\0"
        g["bufferViews"].append({"buffer": 0, "byteOffset": len(buf), "byteLength": arr.nbytes, "target": 34962})
        buf += arr.tobytes()
        g["accessors"].append({"bufferView": len(g["bufferViews"]) - 1, "componentType": 5126, "count": len(arr),
                               "type": "VEC3", "min": arr.min(0).tolist(), "max": arr.max(0).tolist()})
        return len(g["accessors"]) - 1

    mats = {}

    def mat(albedo, metal, rough, emissive=None):
        key = (tuple(np.round(albedo, 4)), metal, rough, tuple(emissive or ()))
        if key not in mats:
            m = {"name": f"isaac_{len(mats)}", "pbrMetallicRoughness": {"baseColorFactor": [*map(float, albedo), 1.0],
                                                                       "metallicFactor": float(metal), "roughnessFactor": float(rough)}}
            if emissive:
                e = np.asarray(emissive, float)
                m["emissiveFactor"] = (e / e.max()).tolist()
                m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": float(e.max())}}
                g.setdefault("extensionsUsed", []).append("KHR_materials_emissive_strength")
            g["materials"].append(m)
            mats[key] = len(g["materials"]) - 1
        return mats[key]

    nodes = g["nodes"]
    for nd in nodes:                                   # robot materials on every mesh under b<idx>_g1_<link>
        m = re.match(r"^b\d+_g1_(.+)$", nd.get("name", ""))
        if not m:
            continue
        mi = mat(*isaac_scene.robot_material("g1/" + m.group(1)))
        stack = list(nd.get("children", [])) + ([nodes.index(nd)] if "mesh" in nd else [])
        while stack:
            c = nodes[stack.pop()]
            if "mesh" in c and not c.get("name", "").startswith("deform"):
                for prim in g["meshes"][c["mesh"]]["primitives"]:
                    prim["material"] = mi
            stack += [k for k in c.get("children", []) if not re.match(r"^b\d+_", nodes[k].get("name", ""))]
    for nd in nodes:                                   # no lights, no stand-in table
        if nd.get("name") == "table":
            nd.pop("mesh", None)
        nd.get("extensions", {}).pop("KHR_lights_punctual", None)
    roots = g["scenes"][g.get("scene", 0)]["nodes"]
    for bx in sc["boxes"]:
        V, F = box_mesh(np.asarray(bx["half"], float))
        N = crease_normals(V, F)
        Rb = np.asarray(bx["R"], float)
        g["meshes"].append({"name": bx["name"].replace("/", "_"), "primitives": [{"attributes": {
            "POSITION": acc(V[F.reshape(-1)] @ Rb.T + bx["center"]), "NORMAL": acc(N @ Rb.T)},
            "material": mat(bx["albedo"], bx["metallic"], bx["roughness"], bx["emissive"]), "mode": 4}]})
        nodes.append({"name": bx["name"].replace("/", "_"), "mesh": len(g["meshes"]) - 1})
        roots.append(len(nodes) - 1)
    g["extensionsUsed"] = sorted(set(g.get("extensionsUsed", [])))
    while len(buf) % 4:
        buf += b"\0"
    g["buffers"][0]["byteLength"] = len(buf)
    g["buffers"][0].pop("uri", None)
    js = json.dumps(g).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(out, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(buf), 0x004E4942)); f.write(bytes(buf))
    Rp, tp = isaac_scene._quat_mat(pel[3:]), pel[:3]
    eye, tgt = Rp @ np.asarray(THIRD["eye_pelvis"]) + tp, Rp @ np.asarray(THIRD["target_pelvis"]) + tp
    z = (eye - tgt) / np.linalg.norm(eye - tgt); x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    Rc = np.stack([x, np.cross(z, x), z], 1)
    Path(out).with_suffix(".scene.json").write_text(json.dumps({"key_travel": sc["key_travel"].tolist(), "third": {
        "pos": eye.tolist(), "quat": _quat_xyzw(Rc), "focal_mm": THIRD["focal_mm"], "aperture_mm": THIRD["aperture_mm"]}}, indent=1))
    names = sorted((nd["name"] for nd in nodes if re.match(r"^b\d+_", nd.get("name", ""))), key=lambda s: int(s[1:s.index("_")]))
    Path(out).with_suffix(".bodies.txt").write_text("\n".join(names) + "\n")
    print(f"[isaacify] {task}: {len(sc['boxes'])} boxes, {len(mats)} Isaac materials, {len(names)} bodies -> {out}")


if __name__ == "__main__":
    main()
