#!/usr/bin/env python3
"""Run RKNN YOLO person detection on a camera (Orange Pi NPU).

Usage (on the board, from kit root):

  python3 examples/infer_camera.py --camera /dev/video0
  python3 examples/infer_camera.py --image path/to.jpg --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yolo_rknn.detector import RknnYoloDetector, print_npu_probe, probe_npu_environment
from yolo_rknn.platform import default_models_dir


def draw_dets(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['label']} {d['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN YOLO camera / image demo")
    parser.add_argument("--camera", default="/dev/video0", help="V4L2 camera device")
    parser.add_argument("--image", default=None, help="Run once on a still image instead of camera")
    parser.add_argument("--model", default=None, help="Path to .rknn (default: auto under models/)")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true", help="Open a preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = forever)")
    args = parser.parse_args()

    probe = probe_npu_environment(default_models_dir(), rknn_model=args.model, imgsz=args.imgsz)
    print_npu_probe(probe)
    if not probe.ready or not probe.rknn_model_path:
        print("NPU not ready. Fix notes above, then retry.", file=sys.stderr)
        return 2

    det = RknnYoloDetector(
        probe.rknn_model_path,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
    )

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: cannot read image {args.image}", file=sys.stderr)
            return 1
        t0 = time.perf_counter()
        dets = det.infer(frame)
        ms = (time.perf_counter() - t0) * 1000
        print(f"detections={len(dets)}  latency={ms:.1f} ms")
        for d in dets:
            print(f"  {d}")
        if args.show:
            draw_dets(frame, dets)
            cv2.imshow("rknn", frame)
            cv2.waitKey(0)
        det.release()
        return 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            dets = det.infer(frame)
            ms = (time.perf_counter() - t0) * 1000
            n += 1
            print(f"[{n}] dets={len(dets)}  {ms:.1f} ms")
            if args.show:
                draw_dets(frame, dets)
                cv2.imshow("rknn", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        det.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
