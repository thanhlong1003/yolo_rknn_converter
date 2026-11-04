"""Standalone YOLO <-> RKNN helpers for Orange Pi 5 (RK3588) and sibling boards."""

from yolo_rknn.detector import RknnYoloDetector
from yolo_rknn.platform import detect_npu_platform, resolve_rknn_model_path

__all__ = [
    "RknnYoloDetector",
    "detect_npu_platform",
    "resolve_rknn_model_path",
]
