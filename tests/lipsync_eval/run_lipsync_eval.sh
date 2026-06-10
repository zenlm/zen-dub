#!/usr/bin/env bash
# Perceptual lip-sync quality eval for a dubbed talking-head video.
#
# Produces:
#   - LSE-C (SyncNet confidence, HIGHER=better) and LSE-D (min distance, LOWER=better)
#     via the canonical syncnet_python (S3FD face-track + crop + SyncNet v2). Eval tooling only.
#   - Numeric artifact metrics with documented pass/fail thresholds:
#       temporal_flicker_excess  (mouth jitter above codec baseline)
#       seam_strength_ratio      (BiSeNet/lower-face paste-edge gradient vs adjacent skin)
#       sharpness_dub_over_source(VAE softening: dub-mouth Laplacian-var / source-mouth, aligned)
#   - Visual inspection artifacts: <prefix>-contact.png (frame contact sheet),
#     <prefix>-mouth.png (mouth-zoom viseme strip + seam overlay),
#     <prefix>-sidebyside.mp4 (source|dub, frame-aligned, with dub audio),
#     <prefix>-sidebyside.png (montage), diff_heatmap.png (where the pipeline writes).
#
# Usage:
#   run_lipsync_eval.sh <dub_video> [source_video] [reference_video]
#     <dub_video>        the dub to evaluate (video+audio). REQUIRED.
#     [source_video]     original undubbed video; enables source-diff bbox recovery,
#                        VAE-softness vs source, and the side-by-side. Pass '-' to skip.
#     [reference_video]  a known-good dub (e.g. PyTorch MuseTalk) to compare LSE against.
#
# Env overrides: OUT (artifact dir, default=video's dir), PREFIX (default=dub basename),
#                EVAL_VENV (default ~/.venv-lipsync-eval), FPS (default 25).
#
# Thresholds are calibrated against the PyTorch-MuseTalk reference dub (see THRESH in
# artifact_metrics.py). The lip-sync PASS bar is: LSE-C within 1.5 of the reference dub
# (or LSE-C >= 4.0 absolute when no reference is given) AND AV offset within +/-3 frames.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUB="${1:?usage: run_lipsync_eval.sh <dub_video> [source_video|-] [reference_video]}"
SRC="${2:-}"; [ "${SRC:-}" = "-" ] && SRC=""
REF="${3:-}"

EVAL_VENV="${EVAL_VENV:-$HOME/.venv-lipsync-eval}"
FPS="${FPS:-25}"
PREFIX="${PREFIX:-$(basename "${DUB%.*}")}"
OUT="${OUT:-$(cd "$(dirname "$DUB")" && pwd)}"
WORK="$HERE/work/run_$PREFIX"
mkdir -p "$WORK" "$OUT"

# ---- bootstrap eval venv (idempotent) ----
if [ ! -x "$EVAL_VENV/bin/python" ]; then
  echo "[setup] creating eval venv at $EVAL_VENV"
  python3 -m venv "$EVAL_VENV"
  "$EVAL_VENV/bin/pip" install -q --upgrade pip
fi
PY="$EVAL_VENV/bin/python"
if ! "$PY" -c "import cv2, scipy, torch, imageio_ffmpeg, python_speech_features, scenedetect" 2>/dev/null; then
  echo "[setup] installing eval deps"
  "$EVAL_VENV/bin/pip" install -q opencv-python-headless numpy scipy imageio-ffmpeg \
      python_speech_features scenedetect tqdm
  "$EVAL_VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
FF="$("$PY" -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"
# expose bundled ffmpeg as 'ffmpeg' for syncnet_python subprocess calls
mkdir -p "$HERE/bin"; ln -sf "$FF" "$HERE/bin/ffmpeg"; export PATH="$HERE/bin:$PATH"

# ---- syncnet weights (idempotent) ----
SP="$HERE/syncnet_python"
if [ ! -d "$SP" ]; then
  echo "[setup] cloning syncnet_python"; git clone --depth 1 https://github.com/joonson/syncnet_python.git "$SP"
fi
if [ ! -s "$SP/data/syncnet_v2.model" ]; then
  echo "[setup] fetching SyncNet weights"
  mkdir -p "$SP/data" "$SP/detectors/s3fd/weights"
  curl -sL --retry 2 -o "$SP/data/syncnet_v2.model" http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model
  curl -sL --retry 2 -o "$SP/detectors/s3fd/weights/sfd_face.pth" https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth
fi

echo "==================================================================="
echo " LIP-SYNC EVAL : $PREFIX"
echo "   dub=$DUB"
echo "   source=${SRC:-<none>}  reference=${REF:-<none>}"
echo "   artifacts -> $OUT"
echo "==================================================================="

# ---- 1. extract frames ----
echo "[1/5] extracting frames @ ${FPS}fps"
"$PY" "$HERE/extract_frames.py" "$DUB" "$WORK/frames_dub" >/dev/null
[ -n "$SRC" ] && "$PY" "$HERE/extract_frames.py" "$SRC" "$WORK/frames_src" >/dev/null || mkdir -p "$WORK/frames_src"

# ---- 2. recover mouth bbox (source-diff if source available, else prior) ----
echo "[2/5] recovering mouth/paste bbox"
"$PY" "$HERE/recover_mouth_bbox.py" "$WORK/frames_dub" "${SRC:+$WORK/frames_src}" "$WORK" >"$WORK/bbox.json"
cat "$WORK/bbox.json"

# ---- 3. SyncNet LSE-C / LSE-D ----
echo "[3/5] SyncNet LSE-C / LSE-D (dub)"
"$PY" "$HERE/run_syncnet_lse.py" "$DUB" "dub_$PREFIX" "$WORK/syncnet" >"$OUT/$PREFIX.lse.json"
cat "$OUT/$PREFIX.lse.json"
if [ -n "$REF" ]; then
  echo "[3/5] SyncNet LSE-C / LSE-D (reference: $(basename "$REF"))"
  "$PY" "$HERE/run_syncnet_lse.py" "$REF" "ref_$PREFIX" "$WORK/syncnet" >"$OUT/$PREFIX.reflse.json"
  cat "$OUT/$PREFIX.reflse.json"
fi

# ---- 4. artifact metrics ----
echo "[4/5] artifact metrics (flicker / seam / VAE softness)"
"$PY" "$HERE/artifact_metrics.py" "$WORK/frames_dub" "$WORK/signal_bbox.json" "${SRC:+$WORK/frames_src}" >"$OUT/$PREFIX.artifacts.json"
cat "$OUT/$PREFIX.artifacts.json"

# ---- 5. visual artifacts ----
echo "[5/5] rendering visual artifacts -> $OUT"
export WORK OUT PREFIX
export DUB_FRAMES="$WORK/frames_dub" SRC_FRAMES="$WORK/frames_src" BBOX="$WORK/signal_bbox.json" AUDIO="$DUB"
"$PY" "$HERE/make_visuals.py"
"$PY" "$HERE/make_mouth.py"
"$PY" "$HERE/make_sidebyside_video.py"
[ -f "$WORK/diff_heatmap.png" ] && cp "$WORK/diff_heatmap.png" "$OUT/$PREFIX-diffheatmap.png"

# ---- summary + verdict ----
echo "==================================================================="
echo " SUMMARY"
"$PY" - "$OUT/$PREFIX.lse.json" "$OUT/$PREFIX.artifacts.json" "${REF:+$OUT/$PREFIX.reflse.json}" <<'PYEOF'
import json,sys
lse=json.load(open(sys.argv[1])); art=json.load(open(sys.argv[2]))
ref=json.load(open(sys.argv[3])) if len(sys.argv)>3 and sys.argv[3] else None
print(f"  LSE-C (confidence, higher=better): {lse.get('lse_c')}")
print(f"  LSE-D (distance,   lower=better):  {lse.get('lse_d')}")
print(f"  AV offset (frames, ~0=best):       {lse.get('av_offset')}")
if ref:
    dc=lse.get('lse_c',0)-ref.get('lse_c',0)
    print(f"  reference LSE-C: {ref.get('lse_c')}  -> native is {dc:+.2f} vs reference")
print("  --- artifact grades ---")
allpass=True
for k,g in art.get('grades',{}).items():
    mark="PASS" if g['pass'] else "FAIL"
    allpass=allpass and g['pass']
    print(f"   [{mark}] {k} = {g['value']} (need {g['op']} {g['threshold']})")
# lip-sync verdict
c=lse.get('lse_c',0); off=abs(lse.get('av_offset',99) or 99)
if ref is not None:
    sync_ok = (c >= ref.get('lse_c',0)-1.5) and off<=3
    basis=f"within 1.5 of ref ({ref.get('lse_c')}) and |offset|<=3"
else:
    sync_ok = c>=4.0 and off<=3
    basis="LSE-C>=4.0 and |offset|<=3"
print(f"  --- VERDICT ---")
print(f"   lip-sync: {'GOOD' if sync_ok else 'WEAK'}  ({basis})")
print(f"   artifacts: {'CLEAN' if allpass else 'DEFECTS PRESENT'}")
PYEOF
echo "==================================================================="
echo " Artifacts written to $OUT :"
ls -1 "$OUT/$PREFIX"-*.png "$OUT/$PREFIX"-*.mp4 "$OUT/$PREFIX".*.json 2>/dev/null | sed 's/^/   /'
