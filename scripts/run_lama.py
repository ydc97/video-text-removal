#!/usr/bin/env python3
"""Fast single-frame text removal with LaMa (TorchScript).

Fills the text bbox with a solid-rect mask per frame, then applies a
low-frequency tone match so the fill blends with the surrounding footage.
Best for smooth/bright backgrounds. If dark textured scenery passes behind
the text, prefer run_propainter.py (per-frame LaMa can leave a smooth light
"capsule" artifact there).
"""
import argparse, os, subprocess, sys
import cv2
import numpy as np
import torch


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", required=True, help="text bbox x0,y0,x1,y1 (frame coords), generous: glyph cores + outline + ~4px")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-margin", type=int, default=110,
                    help="context margin around the bbox for the inpainting window")
    ap.add_argument("--no-lf-match", action="store_true",
                    help="disable the low-frequency tone match pass")
    ap.add_argument("--weights-dir",
                    default=os.environ.get("VTR_WEIGHTS_DIR",
                            os.path.expanduser("~/.cache/video-text-removal/weights")))
    args = ap.parse_args()

    bx0, by0, bx1, by1 = map(int, args.bbox.split(","))
    m = args.crop_margin
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cx0 = max(0, (bx0 - m) // 16 * 16); cy0 = max(0, (by0 - m) // 16 * 16)
    cx1 = min(W, (bx1 + m + 15) // 16 * 16); cy1 = min(H, (by1 + m + 15) // 16 * 16)
    CH, CW = cy1 - cy0, cx1 - cx0

    mask = np.zeros((CH, CW), np.float32)
    mask[by0 - cy0:by1 - cy0, bx0 - cx0:bx1 - cx0] = 1.0
    t_mask = torch.from_numpy(mask)[None, None]

    torch.set_num_threads(os.cpu_count() or 8)
    model = torch.jit.load(os.path.join(args.weights_dir, "lama.pt"), map_location="cpu")
    model.eval()

    w = np.clip(cv2.GaussianBlur(mask, (0, 0), 4), 0, 1)  # LF-match application weight
    nontext = 1.0 - mask
    M = 25                                                # LF-match region margin
    RY0, RY1 = max(0, by0 - M), min(H, by1 + M)
    RX0, RX1 = max(0, bx0 - M), min(W, bx1 + M)
    keep = (1.0 - mask[RY0 - cy0:RY1 - cy0, RX0 - cx0:RX1 - cx0])

    enc = subprocess.Popen(
        [ffmpeg_exe(), "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-i", args.video, "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-c:a", "copy", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)

    i = 0
    t0 = cv2.getTickCount()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        with torch.no_grad():
            out = model(image=torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))[None],
                        mask=t_mask)
        res = (np.clip(out[0].numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        frame[cy0:cy1, cx0:cx1] = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        if not args.no_lf_match:
            roi = frame[RY0:RY1, RX0:RX1].astype(np.float32)
            bg_lf = cv2.GaussianBlur(roi * keep[..., None], (0, 0), 25) / \
                    np.maximum(cv2.GaussianBlur(keep, (0, 0), 25), 0.05)[..., None]
            out_lf = cv2.GaussianBlur(roi, (0, 0), 25)
            delta = bg_lf - out_lf
            roi = roi + delta * w[RY0 - cy0:RY1 - cy0, RX0 - cx0:RX1 - cx0][..., None]
            frame[RY0:RY1, RX0:RX1] = np.clip(roi, 0, 255).astype(np.uint8)

        enc.stdin.write(frame.tobytes())
        i += 1
        if i % 100 == 0:
            el = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
            print(f"{i} frames, {el:.0f}s", flush=True)

    cap.release(); enc.stdin.close(); enc.wait()
    print(f"DONE {i} frames -> {args.out}")


if __name__ == "__main__":
    main()
