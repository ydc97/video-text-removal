#!/usr/bin/env python3
"""Build a static text mask for video inpainting.

Modes:
  rect  --bbox x0,y0,x1,y1            solid rectangle over the text (reliable default)
  auto  --video V --search x0,y0,x1,y1
        detect static bright text inside a search region via per-pixel temporal
        min/max over sampled frames (text is static, background moves). Best on
        plain-ish backgrounds; always check the emitted overlay preview.
Output: full-frame grayscale PNG (white = hole) + `<out>_preview.png` overlay.
"""
import argparse, subprocess, sys
import cv2
import numpy as np


def sample_frames(video, step):
    cap = cv2.VideoCapture(video)
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("rect")
    r.add_argument("--frame-size", required=True, help="WxH of the video")
    r.add_argument("--bbox", required=True, help="x0,y0,x1,y1 in frame coords")
    r.add_argument("--out", required=True)

    a = sub.add_parser("auto")
    a.add_argument("--video", required=True)
    a.add_argument("--search", required=True, help="x0,y0,x1,y1 search region")
    a.add_argument("--sample-step", type=int, default=4)
    a.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.mode == "rect":
        W, H = map(int, args.frame_size.lower().split("x"))
        x0, y0, x1, y1 = map(int, args.bbox.split(","))
        mask = np.zeros((H, W), np.uint8)
        mask[y0:y1, x0:x1] = 255
        cv2.imwrite(args.out, mask)
        print(f"rect mask -> {args.out} ({(x1-x0)*(y1-y0)} px)")
        return

    x0, y0, x1, y1 = map(int, args.search.split(","))
    frames = sample_frames(args.video, args.sample_step)
    if len(frames) < 8:
        print("too few frames sampled", file=sys.stderr); sys.exit(1)
    stack = np.stack(frames)
    gmin = stack.min(0); gmax = stack.max(0); gmed = np.median(stack, 0).astype(np.uint8)

    zone = slice(y0, y1), slice(x0, x1)
    mn = gmin[zone].astype(np.int16); mx = gmax[zone].astype(np.int16)
    white = (mn >= 240).astype(np.uint8)                       # static bright cores
    near = cv2.dilate(white, np.ones((9, 9), np.uint8))
    dark = ((mx <= 45) & (near > 0)).astype(np.uint8)          # static dark outlines
    u = cv2.bitwise_or(white, dark)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    u = cv2.dilate(u, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))

    mask = np.zeros(gmin.shape, np.uint8)
    mask[zone] = u * 255
    cv2.imwrite(args.out, mask)

    # coverage check: bright text pixels in the median that the mask missed
    missed = int(((gmed[zone] >= 238) & (u == 0)).sum())
    vis = cv2.cvtColor(gmed, cv2.COLOR_GRAY2BGR)
    vis[mask > 0] = (0.45 * vis[mask > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    prev = args.out.rsplit(".", 1)[0] + "_preview.png"
    cv2.imwrite(prev, vis)
    print(f"mask px={int((mask>0).sum())}  median-bright missed={missed}  preview={prev}")
    if missed > 0:
        print("WARNING: mask does not cover all static bright pixels - widen thresholds or use rect mode",
              file=sys.stderr)


if __name__ == "__main__":
    main()
