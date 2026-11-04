"""YOLO person detector on Rockchip RKNN NPU (Orange Pi 5 / RK3588 and siblings).

Post-process follows airockchip ``rknn_model_zoo`` YOLOv8 (1 tensor or 3-branch).
Input is UINT8 HWC RGB letterboxed (no /255), same as the zoo demo.

Export on an x86 Linux PC — platform must match the board:

  bash scripts/export_on_pc.sh
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from yolo_rknn.platform import (
    board_label,
    default_models_dir,
    default_rknn_model_name,
    detect_npu_platform,
    export_hint,
    platform_from_model_path,
    resolve_rknn_model_path,
)


@dataclass
class NpuProbeResult:
    soc: str
    npu_platform: str | None
    npu_driver_seen: bool
    rknnlite_import_ok: bool
    librknnrt_found: bool
    librknnrt_paths: list[str]
    rknn_model_path: str | None
    model_platform: str | None
    platform_match: bool
    ready: bool
    notes: list[str]


def _find_librknnrt() -> list[str]:
    hits: list[str] = []
    roots = [
        "/usr/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/local/lib",
        str(Path.home() / ".local/lib"),
    ]
    for env_key in ("LD_LIBRARY_PATH", "RKNN_LITE_LIB_PATH"):
        val = os.environ.get(env_key, "")
        if val:
            roots.extend(val.split(":"))
    seen: set[str] = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        p = Path(root)
        if not p.is_dir():
            continue
        for name in ("librknnrt.so", "librknnrt.so.1"):
            f = p / name
            if f.is_file():
                hits.append(str(f.resolve()))
    return hits


def _dmesg_has_rknpu() -> bool:
    try:
        out = subprocess.run(
            ["dmesg"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        return "RKNPU" in (out.stdout or "")
    except Exception:
        return False


def probe_npu_environment(
    models_dir: Path | None = None,
    rknn_model: str | None = None,
    imgsz: int = 320,
) -> NpuProbeResult:
    models = models_dir or default_models_dir()
    notes: list[str] = []
    npu_platform, soc = detect_npu_platform()
    if npu_platform is None:
        notes.append(f"Unknown Rockchip SoC {soc!r}; cannot auto-select RKNN platform.")

    rknnlite_ok = False
    try:
        import rknnlite  # noqa: F401

        rknnlite_ok = True
    except ImportError:
        notes.append(
            "Python package rknnlite missing. Install rknn-toolkit-lite2 "
            "(aarch64 wheel from github.com/airockchip/rknn-toolkit2). "
            "Run: bash scripts/setup_on_orangepi.sh"
        )

    libs = _find_librknnrt()
    if not libs:
        notes.append(
            "librknnrt.so not found under /usr/lib or LD_LIBRARY_PATH. "
            "Run: bash scripts/setup_on_orangepi.sh"
        )

    npu_drv = _dmesg_has_rknpu()
    if not npu_drv:
        notes.append("RKNPU not seen in dmesg (driver may still load later).")

    model = resolve_rknn_model_path(models, explicit=rknn_model, imgsz=imgsz)
    model_platform = platform_from_model_path(model) if model else None
    platform_match = True

    if npu_platform and model is None:
        expected = default_rknn_model_name(imgsz, npu_platform)
        notes.append(
            f"No .rknn for this board ({board_label(npu_platform)} / {npu_platform}). "
            f"On an x86 PC: {export_hint(npu_platform, imgsz)} "
            f"or bash scripts/export_on_pc.sh — then place {expected} under models/"
        )
        platform_match = False
    elif npu_platform and model_platform and model_platform != npu_platform:
        expected = default_rknn_model_name(imgsz, npu_platform)
        notes.append(
            f"RKNN model targets {model_platform.upper()} but board is {npu_platform.upper()} "
            f"({board_label(npu_platform)}). Export: {export_hint(npu_platform, imgsz)} -> {expected}"
        )
        platform_match = False
    elif model is None:
        notes.append("No .rknn model found under models/. On PC: bash scripts/export_on_pc.sh")
        platform_match = False

    ready = rknnlite_ok and bool(libs) and model is not None and platform_match
    return NpuProbeResult(
        soc=soc,
        npu_platform=npu_platform,
        npu_driver_seen=npu_drv,
        rknnlite_import_ok=rknnlite_ok,
        librknnrt_found=bool(libs),
        librknnrt_paths=libs,
        rknn_model_path=str(model) if model else None,
        model_platform=model_platform,
        platform_match=platform_match,
        ready=ready,
        notes=notes,
    )


def print_npu_probe(probe: NpuProbeResult) -> None:
    label = board_label(probe.npu_platform) if probe.npu_platform else "Rockchip"
    print(f"=== NPU / RKNN probe ({label}) ===")
    print(f"  soc:              {probe.soc}")
    print(f"  npu platform:     {probe.npu_platform or '(unknown)'}")
    print(f"  RKNPU in dmesg:   {probe.npu_driver_seen}")
    print(f"  rknnlite import:  {probe.rknnlite_import_ok}")
    print(f"  librknnrt:        {probe.librknnrt_found}")
    for p in probe.librknnrt_paths[:3]:
        print(f"    - {p}")
    print(f"  rknn model:       {probe.rknn_model_path or '(not found)'}")
    if probe.model_platform:
        print(f"  model platform:   {probe.model_platform}")
    print(f"  platform match:   {probe.platform_match}")
    print(f"  ready for infer:  {probe.ready}")
    for n in probe.notes:
        print(f"  note: {n}")


def letterbox_bgr(
    im_bgr: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox BGR image to (width, height). Returns image, scale, (pad_w, pad_h)."""
    h, w = im_bgr.shape[:2]
    tw, th = int(new_shape[0]), int(new_shape[1])
    scale = min(th / max(h, 1), tw / max(w, 1))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(im_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (tw - nw) / 2.0
    pad_h = (th - nh) / 2.0
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, scale, (pad_w, pad_h)


def _dfl_numpy(position: np.ndarray) -> np.ndarray:
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = y - y.max(axis=2, keepdims=True)
    exp = np.exp(y)
    y = exp / np.maximum(exp.sum(axis=2, keepdims=True), 1e-9)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def _box_process(position: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([imgsz[1] / grid_h, imgsz[0] / grid_w], dtype=np.float32).reshape(1, 2, 1, 1)
    position = _dfl_numpy(position.astype(np.float32))
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def _post_process_yolov8_single_output(
    output: np.ndarray,
    *,
    conf_thresh: float,
    iou_thresh: float,
    person_only: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    pred = np.asarray(output)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise RuntimeError(f"unsupported single-output RKNN tensor shape: {tuple(np.asarray(output).shape)}")

    if pred.shape[0] <= 256 and pred.shape[1] > 256:
        pred = pred.transpose(1, 0)
    elif pred.shape[1] < 5:
        raise RuntimeError(f"unexpected YOLOv8 channel count in RKNN output: {tuple(pred.shape)}")

    if pred.shape[1] < 5:
        raise RuntimeError(f"unexpected YOLOv8 output shape after transpose: {tuple(pred.shape)}")

    boxes_xywh = pred[:, :4].astype(np.float32, copy=False)
    class_scores = pred[:, 4:].astype(np.float32, copy=False)
    if class_scores.shape[1] == 0:
        raise RuntimeError(f"single-output RKNN tensor has no class scores: {tuple(pred.shape)}")

    classes = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)
    mask = scores >= conf_thresh
    if person_only:
        mask &= classes == 0

    boxes_xywh = boxes_xywh[mask]
    classes = classes[mask]
    scores = scores[mask]
    if boxes_xywh.shape[0] == 0:
        return None, None, None

    boxes = np.empty_like(boxes_xywh, dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    nboxes: list[np.ndarray] = []
    nclasses: list[np.ndarray] = []
    nscores: list[np.ndarray] = []
    for c in np.unique(classes):
        inds = np.where(classes == c)[0]
        keep = _nms_boxes(boxes[inds], scores[inds], iou_thresh)
        if keep:
            nboxes.append(boxes[inds][keep])
            nclasses.append(classes[inds][keep])
            nscores.append(scores[inds][keep])
    if not nboxes:
        return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def _post_process_yolov8_rknn(
    input_data: list[np.ndarray],
    *,
    imgsz: tuple[int, int],
    conf_thresh: float,
    iou_thresh: float,
    person_only: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if len(input_data) == 1:
        return _post_process_yolov8_single_output(
            input_data[0],
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            person_only=person_only,
        )

    branches = 3
    if len(input_data) < branches * 2:
        raise RuntimeError(
            f"unexpected RKNN outputs ({len(input_data)} tensors); "
            "expected either 1 raw YOLOv8 tensor or rknn_model_zoo layout (>=6 tensors)"
        )
    pair_per_branch = len(input_data) // branches
    boxes_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    cls_list: list[np.ndarray] = []
    for i in range(branches):
        boxes_list.append(_box_process(input_data[pair_per_branch * i], imgsz))
        cls_list.append(input_data[pair_per_branch * i + 1])
        scores_list.append(
            np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32)
        )

    def _flatten(t: np.ndarray) -> np.ndarray:
        ch = t.shape[1]
        return t.transpose(0, 2, 3, 1).reshape(-1, ch)

    boxes = np.concatenate([_flatten(v) for v in boxes_list])
    classes_conf = np.concatenate([_flatten(v) for v in cls_list])
    obj_conf = np.concatenate([_flatten(v) for v in scores_list]).reshape(-1)

    class_max = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)
    scores = class_max * obj_conf
    mask = scores >= conf_thresh
    if person_only:
        mask &= classes == 0
    boxes = boxes[mask]
    classes = classes[mask]
    scores = scores[mask]
    if boxes.shape[0] == 0:
        return None, None, None

    nboxes: list[np.ndarray] = []
    nclasses: list[np.ndarray] = []
    nscores: list[np.ndarray] = []
    for c in np.unique(classes):
        inds = np.where(classes == c)[0]
        keep = _nms_boxes(boxes[inds], scores[inds], iou_thresh)
        if keep:
            nboxes.append(boxes[inds][keep])
            nclasses.append(classes[inds][keep])
            nscores.append(scores[inds][keep])
    if not nboxes:
        return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def _scale_boxes_to_orig(
    boxes: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out[:, :4] /= max(scale, 1e-9)
    h, w = orig_shape
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 2] = np.clip(out[:, 2], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    out[:, 3] = np.clip(out[:, 3], 0, h - 1)
    return out


class RknnYoloDetector:
    """RKNN YOLO detector returning bbox dicts: x1,y1,x2,y2,confidence,label."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        imgsz: int = 320,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        person_only: bool = True,
        verbose: bool = False,
    ) -> None:
        from rknnlite.api import RKNNLite

        npu_platform, soc = detect_npu_platform()
        model_platform = platform_from_model_path(model_path)
        if npu_platform and model_platform and model_platform != npu_platform:
            raise RuntimeError(
                f"RKNN model is for {model_platform.upper()} but board is {npu_platform.upper()} "
                f"(soc={soc!r}). Export for this board: PLATFORM={npu_platform} bash scripts/export_on_pc.sh"
            )

        self._imgsz = int(imgsz)
        self._imgsz_wh = (self._imgsz, self._imgsz)
        self._conf = float(conf_threshold)
        self._iou = float(iou_threshold)
        self._person_only = bool(person_only)
        self._letterbox_meta: tuple[float, tuple[float, float]] = (1.0, (0.0, 0.0))
        self._rknn: Any = None

        self._rknn = RKNNLite(verbose=verbose)
        ret = self._rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed ret={ret} path={model_path}")
        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed ret={ret}")

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        lb, scale, pad = letterbox_bgr(frame_bgr, self._imgsz_wh)
        self._letterbox_meta = (scale, pad)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)

    def infer(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        inp = self._preprocess(frame_bgr)
        inp = np.expand_dims(inp, axis=0)
        outputs = self._rknn.inference(inputs=[inp])
        if not outputs:
            return []
        outs = [np.asarray(o) for o in outputs]
        boxes, _classes, scores = _post_process_yolov8_rknn(
            outs,
            imgsz=self._imgsz_wh,
            conf_thresh=self._conf,
            iou_thresh=self._iou,
            person_only=self._person_only,
        )
        if boxes is None or scores is None:
            return []
        scale, pad = self._letterbox_meta
        boxes = _scale_boxes_to_orig(boxes, (h, w), scale, pad)
        detections: list[dict[str, Any]] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(scores[i]),
                    "label": "person",
                }
            )
        return detections
