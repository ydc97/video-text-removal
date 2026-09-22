---
name: video-text-removal
description: Remove overlaid text, subtitles, captions, or watermarks from videos using AI video inpainting. Two engines - ProPainter (best quality, cross-frame) and LaMa (fast, single-frame). Use when the user asks to erase/remove/delete text, subtitles (字幕), captions, watermarks, or logos from a video file. Covers text localization, mask creation, inpainting, quality verification, and re-encoding with the original audio preserved.
license: Apache-2.0
---

# Video Text Removal

Eliminate overlaid text (subtitles, captions, watermarks, logos) from a video by
inpainting the covered region with AI, then re-encoding with the original audio.

## One-time setup

Run once per machine before first use (downloads ~380 MB of model weights to a
persistent cache and installs Python dependencies; no GPU required):

```bash
bash "<this-skill-dir>/setup.sh"
```

Verify the environment:

```bash
python3 "<this-skill-dir>/scripts/probe.py" <video.mp4>
```

## Workflow

### Step 1 - Probe the video

```bash
python3 scripts/probe.py <video.mp4>
```

Returns duration, fps, resolution, audio presence. Use fps/duration to estimate
runtime (both engines are roughly linear in frame count).

### Step 2 - Locate the text (requires your eyes, do not skip)

Extract a few sample frames and actually view them:

```bash
ffmpeg -ss <t> -i <video.mp4> -frames:v 1 /tmp/sample_<t>.png
```

Determine the text bounding box (x0, y0, x1, y1) in frame coordinates. Be generous:
include the glyph cores **and** their outlines/antialiasing (+4-6 px on each side).

Confirm the text is **static** (same position every frame) by comparing two frames
from different timestamps. Static text = one bbox for the whole video (the common
case for burned-in subtitles and watermarks). If the text moves, split the video
into segments per position and treat each as a separate job.

### Step 3 - Choose the engine

| Situation | Engine | Typical cost |
|---|---|---|
| Default / best quality / dark or textured background behind text | `run_propainter.py` | ~30 min CPU for 400 frames at 512px crop |
| Fast iteration while tuning the mask, or smooth/bright background behind text | `run_lama.py` | ~6 min CPU for 400 frames |
| Known pure-black region behind text (e.g. letterbox bars) | Either; LaMa is enough | ~6 min |

ProPainter uses cross-frame flow propagation, so its reconstruction is
tone/texture/timing-consistent with the surrounding footage. LaMa is per-frame and
tends to leave a smooth, slightly-too-bright "capsule" artifact when dark, textured
clouds/scenery pass behind the text. If a LaMa result shows that artifact, switch to
ProPainter instead of trying to patch it.

### Step 4 - Run the inpainting

```bash
# Best quality (default choice)
python3 scripts/run_propainter.py --video <video.mp4> --bbox x0,y0,x1,y1 --out clean.mp4

# Fast path
python3 scripts/run_lama.py --video <video.mp4> --bbox x0,y0,x1,y1 --out clean.mp4
```

Both scripts handle cropping, masking, inference, pasting back, and re-encoding
(H.264 + original audio) internally. Add `--crop-margin N` (default ~110) if the
text sits close to other content you want excluded from the processing window.

To build a mask image yourself (e.g. irregular text shapes) instead of using a bbox:

```bash
python3 scripts/build_mask.py rect  --frame-size 1920x1080 --bbox x0,y0,x1,y1 --out mask.png
python3 scripts/build_mask.py auto  --video <video.mp4> --search x0,y0,x1,y1 --out mask.png
```

`auto` detects static bright text via per-pixel temporal min/max over sampled
frames and is useful when the text sits on a plain background. `rect` is the
reliable default.

### Step 5 - Verify (mandatory)

```bash
python3 scripts/verify.py --original <video.mp4> --processed clean.mp4 \
    --band x0,y0,x1,y1 --subject x0,y0,x1,y1
```

- `--band`: the text region. Checks for temporal flicker (must not exceed the
  source's own flicker).
- `--subject`: a box covering any nearby person/face/product that the text was
  close to. Mean abs diff vs the original must be < 5/255 (codec noise level).
  A higher value means the inpainting damaged real content - shrink the mask.

Then visually inspect extracted frames around the worst moments (dark or busy
background behind the text) before delivering. A clean numeric report does not
guarantee a clean freeze-frame.

### Step 6 - Deliver

The output mp4 already contains the original audio track. Deliver alongside the
original and state what was removed and any known residual artifacts.

## Hard-won rules (do not violate)

1. **Never use stroke-level / per-glyph masks with LaMa.** Thin masks and glyph
   outlines make it hallucinate noisy dark mush. Give LaMa a solid rectangle
   (or solid per-word boxes) that fully covers the text plus 4-6 px.
2. **Never use cv2.inpaint (Telea) on strokes crossing a black/light boundary**
   (e.g. letterbox edges) - it bleeds black into the fill, and dark hair/surfaces
   near the text become smears.
3. **Never copy a donor patch from another part of the frame to cover the text**
   without per-column tone matching - the brightness/structure mismatch shows as a
   visible rectangle.
4. **Mask conventions differ**: LaMa (this skill's script) uses float 1.0 = hole;
   ProPainter uses white (255) = hole in a grayscale PNG and dilates the flow mask
   by 4 internally. The scripts handle this - only relevant if you call the models
   directly.
5. **Do not let the mask overlap people/faces/products.** The fill replaces
   whatever is inside it; a rect that reaches a head will erase the head. Measure
   the text extent precisely (view zoomed crops) before choosing the mask bottom.
6. **Keep model weights out of /tmp or other volatile dirs** - the setup script
   defaults to `~/.cache/video-text-removal/weights` (override with
   `VTR_WEIGHTS_DIR`).
7. **A text region that is never revealed in any frame is unknowable.** Both
   engines hallucinate a plausible continuation there. Prefer ProPainter for
   those cases and always disclose residual artifacts to the user.

## Environment notes

- Python 3.9+; dependencies: `torch`, `torchvision`, `opencv-python-headless`,
  `numpy`, `Pillow`, `imageio-ffmpeg`, `einops`, `scipy`, `matplotlib`, `tqdm`.
  `setup.sh` installs them (with PEP 668 `--user --break-system-packages`
  fallback) and pairs torch/torchvision from the official CPU wheel index; GPU
  users may substitute their own CUDA build - ProPainter auto-detects CUDA.
- On Debian/Ubuntu systems without `ensurepip`, bootstrap pip first with
  `python3 <(curl -fsSL https://bootstrap.pypa.io/get-pip.py) --user --break-system-packages`.
- Vendored `third_party/ProPainter` is the official ProPainter inference code
  (Apache-2.0, (c) SCZU). Weights are downloaded by `setup.sh` from their
  official release pages and are licensed by their respective authors.

## Script reference

Run any script with `--help` for full options.

| Script | Purpose |
|---|---|
| `probe.py` | duration / fps / resolution / audio of a video (JSON) |
| `build_mask.py` | rect or auto-detected static-text mask as a PNG |
| `run_lama.py` | fast single-frame inpainting path (LaMa TorchScript + tone matching) |
| `run_propainter.py` | best-quality cross-frame inpainting path (vendored ProPainter) |
| `verify.py` | subject-integrity, band-flicker and residual checks vs the original |
