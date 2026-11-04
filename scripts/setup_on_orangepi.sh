#!/usr/bin/env bash
# Install Rockchip RKNN Lite2 + librknnrt on an aarch64 Orange Pi (host or container).
# Targets Orange Pi 5 (RK3588) and similar boards (RK3566 / RK3568 / RK3576).
#
#   cd /path/to/yolo_rknn_orangepi5
#   sudo bash scripts/setup_on_orangepi.sh
#
# Host prerequisites:
#   - NPU kernel driver loaded (dmesg | grep RKNPU)
#   - If using Docker: --privileged and/or --device /dev/dri
#
# After this script:
#   python3 -m yolo_rknn.check_npu
#   # still need a .rknn exported for YOUR SoC — see scripts/export_on_pc.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RKNN_TAG="${RKNN_TAG:-v2.3.2}"
RKNN_REPO="${RKNN_REPO:-https://github.com/airockchip/rknn-toolkit2.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/rknn-toolkit2}"
PY_BIN="${PY_BIN:-python3}"

detect_rknn_platform() {
  local compat
  compat="$(tr '\0' '\n' < /proc/device-tree/compatible 2>/dev/null || true)"
  if grep -qi 'rk3588' <<<"$compat"; then
    echo rk3588
  elif grep -qi 'rk3576' <<<"$compat"; then
    echo rk3576
  elif grep -qi 'rk3568' <<<"$compat"; then
    echo rk3568
  elif grep -qi 'rk3566' <<<"$compat"; then
    echo rk3566
  else
    echo unknown
  fi
}

NPU_PLATFORM="$(detect_rknn_platform)"
SOC_LINE="$(tr '\0' ' ' < /proc/device-tree/compatible 2>/dev/null | head -c 120 || echo unknown)"

echo "=== RKNN setup on Orange Pi (${NPU_PLATFORM}) ==="
echo "SOC:    ${SOC_LINE}"
echo "Python: $("$PY_BIN" --version 2>&1)"
echo "Arch:   $(uname -m)"
echo "Kit:    $KIT_ROOT"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: this script is for aarch64 Rockchip boards." >&2
  exit 1
fi

if [[ "$NPU_PLATFORM" == "unknown" ]]; then
  echo "WARNING: could not detect RKNN platform from device-tree; set PLATFORM=... when exporting." >&2
fi

if ! "$PY_BIN" -c "import sys; assert sys.version_info[:2] in {(3,8),(3,9),(3,10),(3,11),(3,12)}" 2>/dev/null; then
  echo "ERROR: unsupported Python. RKNN Lite2 supports 3.8–3.12." >&2
  exit 1
fi

PY_TAG="$("$PY_BIN" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
echo "Python tag: cp${PY_TAG}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libopencv-dev \
    python3-opencv \
    python3-numpy \
    file \
    patchelf \
    || true
fi

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  rm -rf "$INSTALL_DIR"
  git clone --depth 1 --branch "$RKNN_TAG" "$RKNN_REPO" "$INSTALL_DIR" \
    || git clone --depth 1 "$RKNN_REPO" "$INSTALL_DIR"
else
  echo "Using existing $INSTALL_DIR"
fi

RUNTIME_DIR="$INSTALL_DIR/rknpu2/runtime/Linux/librknn_api/aarch64"
if [[ ! -d "$RUNTIME_DIR" ]]; then
  echo "ERROR: runtime not found at $RUNTIME_DIR" >&2
  exit 1
fi

echo "Installing librknnrt to /usr/lib ..."
cp -av "$RUNTIME_DIR"/* /usr/lib/
ldconfig

if ! ls /usr/lib/librknnrt.so* >/dev/null 2>&1; then
  echo "ERROR: librknnrt.so not installed under /usr/lib" >&2
  exit 1
fi
ls -la /usr/lib/librknnrt.so*

SERVER_BIN="$INSTALL_DIR/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/rknn_server"
if [[ -x "$SERVER_BIN" ]]; then
  cp -av "$SERVER_BIN" /usr/bin/ 2>/dev/null || true
fi

PKG_DIR="$INSTALL_DIR/rknn-toolkit-lite2/packages"
WHEEL="$(ls "$PKG_DIR"/rknn_toolkit_lite2-*-cp"${PY_TAG}"-cp"${PY_TAG}"-manylinux_2_17_aarch64.manylinux2014_aarch64.whl 2>/dev/null | head -1 || true)"

if [[ -z "$WHEEL" ]]; then
  echo "Wheel not found for cp${PY_TAG} in $PKG_DIR — trying PyPI ..."
  "$PY_BIN" -m pip install --no-cache-dir -U pip
  if ! "$PY_BIN" -m pip install --no-cache-dir "rknn-toolkit-lite2==${RKNN_TAG#v}"; then
    echo "ERROR: no matching wheel for cp${PY_TAG}. Available:" >&2
    ls -1 "$PKG_DIR"/*.whl 2>/dev/null || true
    exit 1
  fi
else
  echo "Installing wheel: $(basename "$WHEEL")"
  "$PY_BIN" -m pip install --no-cache-dir -U pip
  "$PY_BIN" -m pip install --no-cache-dir "$WHEEL"
fi

export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH:-}"
"$PY_BIN" -c "from rknnlite.api import RKNNLite; print('rknnlite OK:', RKNNLite)"

if [[ -f /root/.bashrc ]] && ! grep -q 'RKNN Lite2 runtime' /root/.bashrc 2>/dev/null; then
  cat >>/root/.bashrc <<'EOF'

# RKNN Lite2 runtime (Rockchip NPU)
export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
EOF
fi

MODEL_NAME="best_320_${NPU_PLATFORM}.rknn"
if [[ "$NPU_PLATFORM" == "unknown" ]]; then
  MODEL_NAME="best_320_<platform>.rknn"
fi

echo ""
echo "=== RKNN runtime installed ==="
echo "Detected platform: ${NPU_PLATFORM}"
echo "Next steps:"
echo "  1) On x86 PC: bash scripts/export_on_pc.sh   # default PLATFORM=rk3588"
echo "     # or: PLATFORM=${NPU_PLATFORM} bash scripts/export_on_pc.sh"
echo "  2) Place .rknn at: ${KIT_ROOT}/models/${MODEL_NAME}"
echo "  3) cd ${KIT_ROOT} && python3 -m yolo_rknn.check_npu"
echo "  4) python3 examples/infer_camera.py --camera /dev/video0"
echo ""
echo "NOTE: RK3566 and RK3588 models are NOT interchangeable."
