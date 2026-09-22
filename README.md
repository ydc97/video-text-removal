# video-text-removal

An [Agent Skill](https://agentskills.io) for removing overlaid text — subtitles,
captions, watermarks, logos — from videos with AI inpainting. Works with any agent
that reads SKILL.md (Claude Code, ZCode, and other Agent-Skills-compatible hosts).

Two engines, picked by the agent based on the footage:

- **ProPainter** (vendored, Apache-2.0) — cross-frame flow-propagation inpainting.
  Best quality: reconstruction is tone/texture/timing-consistent with the
  surrounding footage. Runs on CPU (~30 min for 400 frames at a 512 px crop).
- **LaMa** (TorchScript big-lama) — fast single-frame inpainting for iteration and
  smooth/bright backgrounds (~6 min for 400 frames).

The skill encodes a full quality workflow: text localization, mask creation,
engine selection, automated verification gates (subject integrity, temporal
flicker, residual scan), and re-encoding with the original audio preserved.

## Install

```bash
git clone https://github.com/ydc97/video-text-removal.git
cd video-text-removal

# 1) make it available to your agent (optional - agents can also read the folder directly)
./install.sh                 # auto-detects ~/.claude/skills, ~/.zcode/skills, ...
#    or: ./install.sh /path/to/your/agent/skills

# 2) one-time environment setup (weights ~380 MB -> ~/.cache/video-text-removal)
bash setup.sh
```

Requirements: Python 3.9+, `curl`, ~4 GB disk. No GPU needed (CPU works; CUDA is
auto-detected if present).

## Usage

Point your agent at the video and ask it to remove the text. The agent reads
`SKILL.md` and follows the workflow (probe → locate text → choose engine → run →
verify → deliver). The scripts are also usable directly:

```bash
# best quality
python3 scripts/run_propainter.py --video in.mp4 --bbox 795,110,1135,191 --out clean.mp4

# fast
python3 scripts/run_lama.py --video in.mp4 --bbox 795,110,1135,191 --out clean.mp4

# quality gates
python3 scripts/verify.py --original in.mp4 --processed clean.mp4 \
    --band 815,112,1120,190 --subject 950,195,1260,430
```

## How it works

1. **Probe** the video (fps, resolution, audio).
2. **Locate the text**: the agent inspects sample frames and measures the text
   bounding box; for static text on plain backgrounds `build_mask.py auto`
   detects glyphs via per-pixel temporal min/max (text is static while the
   background moves).
3. **Inpaint** the masked region with the chosen engine (ProPainter propagates
   real surrounding content across frames; LaMa reconstructs per frame with a
   low-frequency tone match).
4. **Verify** with numeric gates and visual inspection, then re-deliver.

## Known limits

- A text region that is never revealed in any frame is unknowable - both engines
  synthesize a plausible continuation. Subtle artifacts can remain where dark,
  textured scenery passes behind the text; the verification step catches the bad
  cases.
- Not designed for text that moves around the frame (handle per segment).

## Licenses

- Skill code: Apache-2.0 (see LICENSE)
- Vendored ProPainter inference code: Apache-2.0, (c) SCZU (see NOTICE)
- Model weights (downloaded at setup, not redistributed here) are provided by
  their respective authors: big-lama TorchScript port, ProPainter official
  checkpoints.
