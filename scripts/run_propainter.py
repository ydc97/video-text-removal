#!/usr/bin/env python3
"""Best-quality cross-frame text removal with ProPainter (vendored, CPU-capable).

Pipeline: crop a processing window around the text bbox -> dump frames ->
run vendored ProPainter inference (flow propagation + transformer completion) ->
paste the inpainted window back into full frames -> re-encode with original audio.
"""
import argparse, glob, os, shutil, subprocess, sys, tempfile
import cv2
import numpy as np


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", required=True, help="text bbox x0,y0,x1,y1 (frame coords), generous")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-margin", type=int, default=110,
                    help="context margin around the bbox for the processing window")
    ap.add_argument("--skill-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="skill root (contains third_party/ProPainter)")
    ap.add_argument("--weights-dir",
                    default=os.environ.get("VTR_WEIGHTS_DIR",
                            os.path.expanduser("~/.cache/video-text-removal/weights")))
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    bx0, by0, bx1, by1 = map(int, args.bbox.split(","))
    m = args.crop_margin
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cx0 = max(0, (bx0 - m) // 16 * 16); cy0 = max(0, (by0 - m) // 16 * 16)
    cx1 = min(W, (bx1 + m + 15) // 16 * 16); cy1 = min(H, (by1 + m + 15) // 16 * 16)
    CH, CW = cy1 - cy0, cx1 - cx0
    pp_dir = os.path.join(args.skill_dir, "third_party", "ProPainter")
    inf = os.path.join(pp_dir, "inference_propainter.py")
    if not os.path.isfile(inf):
        print(f"missing {inf} - run setup.sh", file=sys.stderr); sys.exit(1)

    # inference_propainter.py loads weights from ./weights relative to its CWD
    wlink = os.path.join(pp_dir, "weights")
    if not os.path.lexists(wlink):
        os.symlink(os.path.abspath(args.weights_dir), wlink)

    work = tempfile.mkdtemp(prefix="vtr_pp_")
    frames_dir = os.path.join(work, "frames"); os.makedirs(frames_dir)
    mask_path = os.path.join(work, "mask.png")

    # 1) dump cropped frames
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", args.video,
                    "-vf", f"crop={CW}:{CH}:{cx0}:{cy0}", f"{frames_dir}/f_%04d.png"], check=True)
    n = len(glob.glob(f"{frames_dir}/f_*.png"))
    print(f"{n} frames cropped ({CW}x{CH} at {cx0},{cy0})", flush=True)

    # 2) mask (white = hole); ProPainter dilates the flow mask by 4 internally
    mask = np.zeros((CH, CW), np.uint8)
    mask[by0 - cy0:by1 - cy0, bx0 - cx0:bx1 - cx0] = 255
    cv2.imwrite(mask_path, mask)

    # 3) ProPainter inference
    env = dict(os.environ, PYTHONPATH=pp_dir)
    cmd = [sys.executable, inf, "--video", frames_dir, "--mask", mask_path,
           "--output", os.path.join(work, "out"), "--save_frames",
           "--save_fps", str(int(round(fps))), "--subvideo_length", "80",
           "--neighbor_length", "10", "--ref_stride", "10"]
    subprocess.run(cmd, check=True, env=env, cwd=pp_dir)

    out_frames = sorted(glob.glob(os.path.join(work, "out", "frames", "frames", "*.png"))) or \
                 sorted(glob.glob(os.path.join(work, "out", "**", "frames", "*.png"), recursive=True))
    if len(out_frames) != n:
        print(f"expected {n} output frames, got {len(out_frames)}", file=sys.stderr); sys.exit(1)

    # 4) paste back + encode (original audio preserved)
    enc = subprocess.Popen(
        [ffmpeg_exe(), "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-i", args.video, "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-c:a", "copy", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(args.video)
    i = 0
    for pf in out_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame[cy0:cy1, cx0:cx1] = cv2.imread(pf)
        enc.stdin.write(frame.tobytes()); i += 1
    cap.release(); enc.stdin.close(); enc.wait()
    print(f"DONE {i} frames -> {args.out}")

    if not args.keep_intermediate:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"intermediate kept in {work}")


if __name__ == "__main__":
    main()
