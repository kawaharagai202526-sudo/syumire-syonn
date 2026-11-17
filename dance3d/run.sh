#!/usr/bin/env bash
# 動画 → 金髪の人物を特定 → 3D姿勢 → カメラのズーム/パン補正 → 3Dアニメ動画（通常速度・0.5倍速）
#
#   ./dance3d/run.sh input/video.mp4 [出力フォルダ] [track.py への追加オプション...]
#
# 例: 自動判定がうまくいかない時は、10.0秒の時点で画面の横40%・縦50%あたりにいる人、と指定:
#   ./dance3d/run.sh input/video.mp4 output --hint 10.0,0.40,0.50
# 例: 動画の最後に関係ない画面（終了画面など）がある時は、解析するフレーム数を指定:
#   ./dance3d/run.sh input/video.mp4 output --max-frames 5895
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
fetch https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx yolox_s.onnx

"$PY" "$HERE/track.py" "$VIDEO" -o "$OUT/motion.npz" --cache "$OUT/pass1.npz" \
      --debug-video "$OUT/check_target.mp4" "$@"
"$PY" "$HERE/camera.py" "$VIDEO" "$OUT/pass1.npz" -o "$OUT/camera.npz"
"$PY" "$HERE/motion3d.py" "$OUT/motion.npz" -o "$OUT/motion3d.npz" --camera "$OUT/camera.npz"
"$PY" "$HERE/render.py" "$OUT/motion3d.npz" -o "$OUT/dance3d.mp4"
"$PY" "$HERE/render.py" "$OUT/motion3d.npz" -o "$OUT/dance3d_slow.mp4" --speed 0.5 --fps 30
echo "done: $OUT/dance3d.mp4  $OUT/dance3d_slow.mp4  (確認用: $OUT/check_target.mp4)"
