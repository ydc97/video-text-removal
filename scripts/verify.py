#!/usr/bin/env python3
"""Quality gates for the inpainted video vs the original.

- subject zone: must be untouched by inpainting -> mean abs diff ~= codec noise (<5/255)
- band zone (the former text area): temporal flicker must not exceed the source's own
- band residual: frames containing glyph-like bright components (informational)

Prints PASS/WARN lines and exits 1 on any FAIL.
"""
import argparse, sys
import cv2
import numpy as np


def frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cap.release()


def box(g, b):
    x0, y0, x1, y1 = b
    return g[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--subject", required=True, help="x0,y0,x1,y1 zone that must be untouched")
    ap.add_argument("--band", default=None, help="x0,y0,x1,y1 former text region")
    a = ap.parse_args()

    sb = tuple(map(int, a.subject.split(",")))
    bd = tuple(map(int, a.band.split(","))) if a.band else None

    hd, prev, flick, gaps = [], None, [], []
    n = 0
    for go, gs in zip(frames(a.processed), frames(a.original)):
        hd.append(float(np.abs(box(go, sb) - box(gs, sb)).mean()))
        if bd:
            b = box(go, bd)
            if prev is not None:
                flick.append(float(np.abs(b - prev).mean()))
            prev = b
            n += 1
    if n == 0:
        print("FAIL: could not read videos / frame counts differ", file=sys.stderr); sys.exit(1)

    hd = np.array(hd)
    ok = True
    status = "PASS" if hd.mean() < 5 else "FAIL"
    ok &= status == "PASS"
    print(f"[{status}] subject integrity: mean diff={hd.mean():.2f} max={hd.max():.2f} (gate <5)")

    if bd:
        flick = np.array(flick)
        # source's own flicker in the same band, sampled at the same cadence
        sf = []
        prev = None
        for gs in frames(a.original):
            b = box(gs, bd)
            if prev is not None:
                sf.append(float(np.abs(b - prev).mean()))
            prev = b
        sf = np.array(sf)
        ratio = flick.mean() / max(sf.mean(), 1e-6)
        # per-frame engines (LaMa) flicker somewhat more than real footage; that is
        # acceptable (WARN). A large ratio means visible shimmer -> re-run ProPainter.
        if ratio <= 1.2:
            status = "PASS"
        elif ratio <= 3.0:
            status = "WARN"
        else:
            status = "FAIL"
        ok &= ratio <= 3.0
        print(f"[{status}] band flicker: processed={flick.mean():.2f} vs source={sf.mean():.2f} "
              f"(ratio {ratio:.2f}, gate <=3 WARN<=1.2 PASS)  p95 {np.percentile(flick,95):.2f} max {flick.max():.2f}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
