#!/usr/bin/env python3
"""Probe a video: duration, fps, resolution, audio. Prints JSON."""
import argparse, json, re, subprocess, sys


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def probe(path):
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    err = p.stderr
    out = {"path": path, "raw": None}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if m:
        out["duration_s"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)
    m = re.search(r"Stream #.*Video:.*?, (\d+)x(\d+)", err)
    if m:
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+\.?\d*)\s*fps", err)
    if m:
        out["fps"] = float(m.group(1))
    out["audio"] = bool(re.search(r"Stream #.*Audio:", err))
    m = re.search(r"Stream #.*Video:.*?(?:, )?(\w+) \(", err)
    if m:
        out["codec"] = m.group(1)
    if out.get("duration_s") and out.get("fps"):
        out["frames"] = int(round(out["duration_s"] * out["fps"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    a = ap.parse_args()
    info = probe(a.video)
    if "duration_s" not in info:
        print("probe failed:", file=sys.stderr); sys.exit(1)
    print(json.dumps(info, indent=2))
