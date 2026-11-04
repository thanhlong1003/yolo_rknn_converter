#!/usr/bin/env python3
"""CLI: probe Rockchip NPU / RKNN readiness for this kit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as ``python -m yolo_rknn.check_npu`` from the kit root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yolo_rknn.detector import print_npu_probe, probe_npu_environment
from yolo_rknn.platform import default_models_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Orange Pi NPU / RKNN setup")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing .rknn files (default: <kit>/models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit path to a .rknn model",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args(argv)

    models_dir = args.models_dir or default_models_dir()
    probe = probe_npu_environment(models_dir, rknn_model=args.model, imgsz=args.imgsz)
    print_npu_probe(probe)
    if probe.ready:
        return 0
    if not probe.platform_match:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
