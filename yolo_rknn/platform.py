"""Rockchip NPU platform detection and RKNN model path helpers."""

from __future__ import annotations

from pathlib import Path

_PLATFORM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("rk3588", "rk3588"),
    ("rk3576", "rk3576"),
    ("rk3568", "rk3568"),
    ("rk3566", "rk3566"),
)

_BOARD_LABELS: dict[str, str] = {
    "rk3566": "Orange Pi 3B",
    "rk3568": "Orange Pi / RK3568",
    "rk3576": "RK3576",
    "rk3588": "Orange Pi 5 / RK3588",
}


def read_soc_compatible_parts() -> list[str]:
    try:
        raw = Path("/proc/device-tree/compatible").read_bytes()
        return [p.decode("ascii", errors="ignore") for p in raw.split(b"\x00") if p]
    except OSError:
        return []


def detect_npu_platform() -> tuple[str | None, str]:
    """Return ``(rknn_platform, soc_label)`` for the current board."""
    parts = read_soc_compatible_parts()
    joined = " ".join(parts).lower()
    for needle, platform in _PLATFORM_PATTERNS:
        if needle in joined:
            soc = next((p for p in parts if needle in p.lower()), parts[0] if parts else "unknown")
            return platform, soc
    return None, (parts[0] if parts else "unknown")


def board_label(platform: str | None) -> str:
    if platform and platform in _BOARD_LABELS:
        return _BOARD_LABELS[platform]
    return "Rockchip board"


def platform_from_model_path(path: str | Path) -> str | None:
    name = Path(path).name.lower()
    for platform in ("rk3588", "rk3576", "rk3568", "rk3566"):
        if platform in name:
            return platform
    return None


def default_rknn_model_name(imgsz: int, platform: str) -> str:
    return f"best_{imgsz}_{platform}.rknn"


def export_hint(platform: str, imgsz: int = 320) -> str:
    return f"yolo export model=best.pt format=rknn name={platform} imgsz={imgsz}"


def rknn_model_candidates(
    models_dir: Path,
    *,
    imgsz: int = 320,
    platform: str | None = None,
) -> list[Path]:
    """Return candidate RKNN model paths under a flat ``models/`` directory."""
    names: list[str] = []
    if platform:
        names.extend(
            [
                default_rknn_model_name(imgsz, platform),
                f"best_{imgsz}-{platform}.rknn",
                f"best_{platform}.rknn",
            ]
        )
    names.extend(
        [
            "best_320_rk3588.rknn",
            "best_320_rk3566.rknn",
            "best.rknn",
        ]
    )
    seen: set[str] = set()
    out: list[Path] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = models_dir / name
        if p.is_file():
            out.append(p)
    return out


def resolve_rknn_model_path(
    models_dir: Path,
    *,
    explicit: str | None = None,
    imgsz: int = 320,
    platform: str | None = None,
) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None
    detected, _ = detect_npu_platform()
    plat = platform or detected
    candidates = rknn_model_candidates(models_dir, imgsz=imgsz, platform=plat)
    if not candidates:
        return None
    if plat:
        for c in candidates:
            mp = platform_from_model_path(c)
            if mp is None or mp == plat:
                return c
        return None
    return candidates[0]


def tool_root() -> Path:
    """Return the root of this standalone kit (parent of the ``yolo_rknn`` package)."""
    return Path(__file__).resolve().parents[1]


def default_models_dir() -> Path:
    return tool_root() / "models"
