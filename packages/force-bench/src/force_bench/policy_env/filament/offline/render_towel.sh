#!/bin/bash
# render_towel.sh FOLDER OUTDIR: a SIM-DEFORMABLE hand-off folder (scene.glb, rollout_states.npz, towel_points_f32.bin,
# towel_normals_f32.bin) in the Isaac FoldTowel scene with ContactBench render v4's look; head + third person, 30 fps.
# Needs: FORK (a contactbench-sim checkout at g1-filament fca23a4 or later), SERVER (filament_server built by
# $FORK/packages/force-bench/src/force_bench/policy_env/filament/build_server.sh), PY (python with numpy + pillow),
# ffmpeg on PATH.
set -euo pipefail
IN=$(cd "$1" && pwd); OUT=$2; HERE=$(cd "$(dirname "$0")" && pwd)
: "${FORK:?set FORK}"; : "${SERVER:?set SERVER}"; PY=${PY:-python3}
mkdir -p "$OUT"
PYTHONPATH=$FORK/packages/force-bench/src $PY "$HERE/isaacify_glb.py" "$IN/scene.glb" "$IN/rollout_states.npz" fold_towel "$OUT/isaac.glb"
KD=$($PY -c "import json;print(','.join(f'{v:.4f}' for v in json.load(open('$OUT/isaac.scene.json'))['key_travel']))")
# the look (force_bench.policy_env.filament.isaac_scene.LOOK / cameras.server_options): dome 25000 x (0.74,0.76,0.72),
# key 35000 lux (1.0,0.97,0.90) 8 deg disk, EV100 14.8, ACES, SSR, background = the dome
OPTS=(--opt ibl_color=0.74,0.76,0.72 --opt sun_dir=$KD --opt sun_color=1.0,0.97,0.9 --opt sun_radius_deg=4 --opt skybox=1
      --opt ev100=14.8 --opt tonemap=aces --opt ssr=1)
DEF=(); [ -f "$IN/towel_points_f32.bin" ] && DEF=(--deform-points "$IN/towel_points_f32.bin" --deform-normals "$IN/towel_normals_f32.bin")
C=(--server "$SERVER" --ibl-none --lux 25000,35000 "${OPTS[@]}" --glb "$OUT/isaac.glb" --names "$OUT/isaac.bodies.txt"
   --states "$IN/rollout_states.npz" --stride 2 "${DEF[@]}")
# head: FoldTowel's d435 mount (spec section 9: 9.0 mm on 20 mm, stock x rot_x(+4 deg)) on g1/d435_link
$PY "$HERE/replay.py" "${C[@]}" --out "$OUT/h" --size 640x480 --camera head:g1/d435_link --head-focal-mm 9.0 \
    --head-rot-ros 0.51715,-0.48225,0.48225,-0.51715
$PY "$HERE/replay.py" "${C[@]}" --out "$OUT/t" --size 800x600 --camera third:"$OUT/isaac.scene.json"
ffmpeg -v error -y -framerate 30 -i "$OUT/h/cam0/f%05d.png" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/head.mp4"
ffmpeg -v error -y -framerate 30 -i "$OUT/t/cam0/f%05d.png" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/third.mp4"
echo "wrote $OUT/head.mp4 $OUT/third.mp4"
