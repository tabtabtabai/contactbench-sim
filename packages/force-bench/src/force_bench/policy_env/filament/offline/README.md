# Offline render v4: replay a recorded rollout in the Isaac scene

Kinematic replay through `filament_server` (`../server.cpp`, built by `../build_server.sh`): every frame sets each
body from a `rollout_states.npz` (t, body_q [T, B, 7] xyzw, body_labels) and, optionally, one deforming mesh from
per-frame sidecars. Physics is never run. Used for the cloth (towel) proof clips and for re-rendering G1 records.

## Inputs

- `scene.glb`: G1 body nodes `b<idx>_<label>` in `body_labels` order (meshes as children, body-local), plus any
  static nodes (e.g. a rail), plus an optional deforming node `deform/towel` (one indexed primitive with
  POSITION / NORMAL / TEXCOORD_0 and its own material, e.g. KHR_materials_sheen). No table, room or lights:
  `isaacify_glb.py` adds them.
- `rollout_states.npz`, as above.
- `towel_points_f32.bin` and `towel_normals_f32.bin`: [T, Nv, 3] float32, world metres, the deform node's vertex order,
  at the rollout's frames (normals optional: area-weighted normals are computed if absent).

## Run

```bash
export FORK=<this checkout> SERVER=<filament_server> PY=<python with numpy + pillow>   # ffmpeg on PATH
bash render_towel.sh <hand-off folder> <out folder>     # head.mp4 (640x480) + third.mp4 (800x600), 30 fps
```

`render_towel.sh` uses the FoldTowel scene (spec section 9: table centre y 0.47, depth 0.66; head camera 9.0 mm,
stock x rot_x(+4 deg) on `g1/d435_link`). For a ColorArrangement-layout scene pass `color_arrangement` to
`isaacify_glb.py`, and use the harness head (`--head-focal-mm 7.6`, the default rotation).

## The look (`../isaac_scene.py`, shared with the live policy cameras)

- Room and table: `bright_room.usd` + `plain_table.usd` boxes (linear colours), placed by the `g1/pelvis` pose at
  frame 0 (Isaac pelvis (-0.15, 0, 0.76) facing +y).
- Lights: a uniform dome (0.74, 0.76, 0.72) at 25000 (in-process, `ibl_color`; also the background); the key
  (1.0, 0.97, 0.90) at 35000 lux with an 8 deg disk, from the window side 70 deg up.
- Camera: EV100 14.8, ACES, screen-space reflections, SSAO (radius 0.15), PCSS shadows, MSAA 4x + FXAA.
- G1: the Isaac USD's OmniPBR materials per link; body metal albedo x1.6 and Dex1 albedo 0.015 (fitted).
- Fit: exposure, key/dome balance and the two robot scales were fitted to REAL G1 head frames (FoldTowel ep0 frame 0
  and a ColorArrangement frame), by the region means of table, wall, silver arm and gripper: error 20.5 in 8-bit units,
  against 57 for a fit to an Isaac RTX render.
- The towel's own look (base-colour and terry normal textures, sheen, the solidified shell) comes with the hand-off
  glb; colours in it must be authored against the table's LINEAR albedo (0.72, 0.69, 0.61).

## Notes

- `deform/towel`'s static primitive is hidden and replaced by a renderable whose positions and tangents are rebuilt
  every frame; it reuses the node's glTF material. gltfio's ubershader multiplies the base colour by a vertex COLOR
  attribute, so the server supplies white vertex colours (without them the cloth renders black or rainbow).
- The third-person camera is not from the spec (no scored Dex1 task has one): a front-left three-quarter view.
- Cost on an L4: ~7 ms per 640x480 frame at 3,962 cloth vertices (18 ms at 15,122).
