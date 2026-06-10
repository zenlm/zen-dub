# Perceptual lip-sync quality eval

Measures whether a MuseTalk dub **looks good / syncs to audio / is artifact-free** — complementary
to the existing cosine-vs-PyTorch parity tests (which only prove we *match* reference MuseTalk numerically).

This is **eval tooling, not the production pipeline.** SyncNet (the standard lip-sync-error metric)
and S3FD are pulled from the upstream `syncnet_python` reference and run on CPU.

## Quick start

```bash
./run_lipsync_eval.sh <dub_video> [source_video|-] [reference_dub]

# e.g. our native dub, vs its source, compared against the PyTorch-MuseTalk dub:
./run_lipsync_eval.sh \
    /mnt/c/Users/z/OneDrive/Desktop/zen-dub-fullnative.mp4 \
    ~/work/zen/gh/zen-dub/data/video/sun.mp4 \
    /mnt/c/Users/z/OneDrive/Desktop/zen-dub-demo.mp4
```

First run bootstraps a venv (`~/.venv-lipsync-eval`), clones `syncnet_python`, and downloads the
SyncNet v2 + S3FD weights (~140 MB, from robots.ox.ac.uk). Subsequent runs are incremental.
The host needs **no system ffmpeg** — a static ffmpeg is provided by the `imageio-ffmpeg` wheel.

## What it produces

Written to `OUT` (default = the dub's directory), prefixed with the dub basename:

| Artifact | What it shows |
|---|---|
| `<p>-contact.png`     | frame contact sheet across the whole clip (global sanity) |
| `<p>-mouth.png`       | mouth-zoom viseme strip (top) + paste-region overlay (bottom) — read visemes & seam |
| `<p>-sidebyside.mp4`  | source \| dub, frame-aligned through the loop, with the dub audio — seams pop in motion |
| `<p>-sidebyside.png`  | static side-by-side montage |
| `<p>-diffheatmap.png` | accumulated source−dub diff — exactly where the pipeline writes (the mouth) |
| `<p>.lse.json`        | LSE-C / LSE-D / AV-offset for the dub |
| `<p>.reflse.json`     | same for the reference dub (if given) |
| `<p>.artifacts.json`  | numeric artifact metrics + pass/fail grades |

## Metrics

**LSE-C** (SyncNet confidence, **higher = better**) and **LSE-D** (min distance, **lower = better**)
via the canonical `syncnet_python` pipeline (S3FD face-track → 224px mouth crop → SyncNet v2,
MFCC audio vs visual embedding, ±15-frame search). `AV offset` is the audio→video lag in frames
that maximizes correlation (~0 = well-synced). Validated against the upstream `example.avi`
(reproduces conf 8.327 / dist 6.613).

**Artifact metrics** (`artifact_metrics.py`), thresholds **calibrated against the PyTorch-MuseTalk
reference dub** so a PASS means "no worse than reference":

| Metric | Meaning | Threshold |
|---|---|---|
| `temporal_flicker_excess`     | mean mouth frame-to-frame abs-diff (0–255) minus a static-region (forehead) codec-noise baseline | **≤ 4.0** (PyTorch ref ≈ 1.1) |
| `seam_strength_ratio`         | gradient magnitude on the lower-face/BiSeNet paste edge ÷ adjacent untouched-skin gradient (≈1 = invisible) | **≤ 3.0** (PyTorch ref ≈ 2.55) |
| `sharpness_dub_over_source`   | dub-mouth Laplacian-variance ÷ **source**-mouth Laplacian-variance on aligned frames — isolates VAE softening | **≥ 0.70** (1.0 = matches original) |

**Lip-sync verdict:** GOOD iff `LSE-C` within 1.5 of the reference dub (or ≥ 4.0 absolute when no
reference) **and** `|AV offset| ≤ 3` frames.

## Mouth bbox recovery

The dub modifies only the lower-face/mouth paste region, so `recover_mouth_bbox.py` localizes it
**empirically** by accumulating the source−dub abs-diff over loop-aligned frame pairs (no need for
the SFD+FAN/DWPose weights, which aren't on disk). The hot blob is the mouth; its bbox/center drive
the mouth strip and the artifact ROIs. Falls back to a lower-center prior (frac 0.48, 0.375) when no
source is supplied. The dub loops its source as a ~150-frame (6 s) forward/backward **boomerang**
repeated ~4× to cover ~24 s of TTS; alignment uses nearest-upper-face matching, which handles the
ping-pong automatically, so sync is evaluated within the looped segments.

## Files

- `run_lipsync_eval.sh`     — orchestrator (the repeatable entry point)
- `run_syncnet_lse.py`      — S3FD track + crop + SyncNet → LSE-C/LSE-D JSON
- `artifact_metrics.py`     — flicker / seam / VAE-softness with graded thresholds
- `recover_mouth_bbox.py`   — empirical mouth-bbox recovery + diff heatmap
- `extract_frames.py`       — ffmpeg frame dump
- `make_visuals.py`         — contact sheet + side-by-side montage
- `make_mouth.py`           — mouth-zoom viseme strip + seam overlay
- `make_sidebyside_video.py`— source|dub aligned mp4 with audio
- `syncnet_python/`         — upstream reference (cloned; weights gitignored)
