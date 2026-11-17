#!/usr/bin/env bash
# 動画 → 金髪の人物を特定 → 3D姿勢 → 3Dアニメ動画（通常速度・0.5倍速）
#
#   ./dance3d/run.sh input/video.mp4 [出力フォルダ] [track.py への追加オプション...]
#
# 例: 自動判定がうまくいかない時は、10.0秒の時点で画面の横40%・縦50%あたりにいる人、と指定:
#   ./dance3d/run.sh input/video.mp4 output --hint 10.0,0.40,0.50
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VIDEO=${1:?usage: run.sh VIDEO [OUTDIR] [track.py options...]}
OUT=${2:-$HERE/../output}
shift; [ $# -gt 0 ] && shift
PY=${PYTHON:-python3}

mkdir -p "$HERE/models" "$OUT"
M=https://storage.googleapis.com/mediapipe-models
fetch() { [ -s "$HERE/models/$2" ] || curl -fsSL -o "$HERE/models/$2" "$1"; }
fetch $M/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task pose_landmarker_full.task
fetch $M/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task pose_landmarker_heavy.task
fetch $M/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite hair_segmenter.tflite

"$PY" "$HERE/track.py" "$VIDEO" -o "$OUT/motion.npz" --cache "$OUT/pass1.npz" \
      --debug-video "$OUT/check_target.mp4" "$@"
"$PY" "$HERE/motion3d.py" "$OUT/motion.npz" -o "$OUT/motion3d.npz"
"$PY" "$HERE/render.py" "$OUT/motion3d.npz" -o "$OUT/dance3d.mp4"
"$PY" "$HERE/render.py" "$OUT/motion3d.npz" -o "$OUT/dance3d_slow.mp4" --speed 0.5 --fps 30
echo "done: $OUT/dance3d.mp4  $OUT/dance3d_slow.mp4  (確認用: $OUT/check_target.mp4)"
