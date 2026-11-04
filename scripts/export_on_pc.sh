#!/usr/bin/env bash
# Export YOLO best.pt -> RKNN for Orange Pi 5 (RK3588) on an x86 Linux PC.
#
# This tool is standalone. Default PLATFORM=rk3588.
#
# Common mistakes this script avoids:
#   - pip install rknn-toolkit2          # NOT on PyPI — use GitHub wheel
#   - pip install rknn-toolkit-lite2 on PC  # lite2 is aarch64 inference ONLY
#   - Python 3.13 system env             # Rockchip wheels stop at cp312
#
# Usage (from kit root on PC):
#   bash scripts/export_on_pc.sh
#
# Or with explicit paths:
#   bash scripts/export_on_pc.sh /path/to/best.pt /path/to/out/best_320_rk3588.rknn
#
# Env overrides: PLATFORM IMGSZ DTYPE RKNN_TAG MODEL_DIR VENV_DIR CACHE_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$KIT_ROOT/models}"
PT_MODEL="${1:-$MODEL_DIR/best.pt}"
RKNN_TAG="${RKNN_TAG:-v2.3.2}"
IMGSZ="${IMGSZ:-320}"
DTYPE="${DTYPE:-fp}"
VENV_DIR="${VENV_DIR:-$KIT_ROOT/.venv-rknn}"
CACHE_DIR="${CACHE_DIR:-$KIT_ROOT/.cache/rknn-export}"

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
    echo ""
  fi
}

if [[ -n "${PLATFORM:-}" ]]; then
  :
elif detected="$(detect_rknn_platform)" && [[ -n "$detected" ]]; then
  PLATFORM="$detected"
else
  PLATFORM="rk3588"
fi

OUT_RKNN="${2:-$MODEL_DIR/best_${IMGSZ}_${PLATFORM}.rknn}"

echo "=== YOLO -> RKNN export (PC x86) ==="
echo "PT:       $PT_MODEL"
echo "OUT:      $OUT_RKNN"
echo "platform: $PLATFORM  imgsz: $IMGSZ  dtype: $DTYPE"

if [[ ! -f "$PT_MODEL" ]]; then
  echo "ERROR: weights not found: $PT_MODEL" >&2
  echo "Place a YOLO .pt file under models/best.pt or pass the path as arg 1." >&2
  exit 1
fi

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "ERROR: python3.10 not found. Install python3.10, python3.10-venv, python3.10-dev first." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating local venv: $VENV_DIR"
  python3.10 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

echo "Using: $(python --version) at $(which python)"

WORKDIR="${WORKDIR:-$CACHE_DIR/work}"
mkdir -p "$WORKDIR" "$CACHE_DIR" "$MODEL_DIR"
cd "$WORKDIR"

if [[ ! -d rknn-toolkit2/.git ]]; then
  rm -rf rknn-toolkit2
  echo "Cloning rknn-toolkit2 ..."
  git clone --depth 1 --branch "$RKNN_TAG" https://github.com/airockchip/rknn-toolkit2.git \
    || git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git
else
  echo "Reusing cached rknn-toolkit2: $WORKDIR/rknn-toolkit2"
fi

PYTAG="$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
RKNN_PKG="rknn-toolkit2/rknn-toolkit2/packages/x86_64"
REQ="${RKNN_PKG}/requirements_cp${PYTAG}-${RKNN_TAG#v}.txt"
WHEEL=("${RKNN_PKG}"/rknn_toolkit2-*-cp"${PYTAG}"-cp"${PYTAG}"-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)

if [[ ! -f "$REQ" ]]; then
  echo "ERROR: no requirements file for cp${PYTAG}. Use Python 3.10–3.12." >&2
  exit 1
fi

echo "Installing rknn-toolkit2 dependencies ..."
python -m pip install -U pip
python -m pip install --no-user "setuptools>=65,<81"
python -m pip install --no-user -r "$REQ"
python -m pip install --no-user "${WHEEL[0]}"
python -m pip install --no-user --force-reinstall "nvidia-nvjitlink-cu12==12.1.105"
python -m pip install --no-user "torch<=2.4.0,>=1.10.1" "torchvision<=0.19.1,>=0.15.0"

python -c "from rknn.api import RKNN; print('rknn-toolkit2 OK')"

ONNX_MODEL="$WORKDIR/best_${IMGSZ}.onnx"
if [[ -f "$ONNX_MODEL" ]]; then
  echo "Reusing cached ONNX: $ONNX_MODEL"
elif [[ -f "${PT_MODEL%.pt}.onnx" ]] && [[ "$(realpath "${PT_MODEL%.pt}.onnx")" != "$ONNX_MODEL" ]]; then
  cp "${PT_MODEL%.pt}.onnx" "$ONNX_MODEL"
  echo "Reusing existing ONNX: ${PT_MODEL%.pt}.onnx"
elif [[ -f "$MODEL_DIR/best_${IMGSZ}.onnx" ]]; then
  cp "$MODEL_DIR/best_${IMGSZ}.onnx" "$ONNX_MODEL"
  echo "Reusing kit ONNX: $MODEL_DIR/best_${IMGSZ}.onnx"
else
  echo "Installing ultralytics for ONNX export (keeps torch<=2.4) ..."
  python -m pip install --no-user "ultralytics>=8.3,<8.4" "onnx>=1.12,<1.19" onnxslim onnxruntime
  python -m pip install --no-user --force-reinstall "torch==2.4.0" "torchvision==0.19.0" \
    "nvidia-nvjitlink-cu12==12.1.105" "setuptools>=65,<81"
  echo "Exporting ONNX imgsz=$IMGSZ ..."
  yolo export model="$PT_MODEL" format=onnx imgsz="$IMGSZ" simplify=True
  cp "$(dirname "$PT_MODEL")/$(basename "${PT_MODEL%.pt}.onnx")" "$ONNX_MODEL" 2>/dev/null \
    || cp "$(basename "${PT_MODEL%.pt}.onnx")" "$ONNX_MODEL"
fi

python -m pip install --no-user "setuptools>=65,<81"
python -c "from rknn.api import RKNN; import pkg_resources; print('convert deps OK')"

if [[ ! -d rknn_model_zoo ]]; then
  echo "Cloning rknn_model_zoo ..."
  git clone --depth 1 https://github.com/airockchip/rknn_model_zoo.git
else
  echo "Reusing cached rknn_model_zoo: $WORKDIR/rknn_model_zoo"
fi

CONVERT_PY="$(find rknn_model_zoo -path '*/yolov8/python/convert.py' | head -1)"
if [[ -z "$CONVERT_PY" ]]; then
  echo "ERROR: convert.py not found in rknn_model_zoo" >&2
  exit 1
fi

echo "Converting ONNX -> RKNN ($DTYPE) ..."
(
  cd "$(dirname "$CONVERT_PY")"
  if [[ "$DTYPE" == "i8" ]]; then
    mkdir -p ../../../datasets/COCO
    if [[ ! -f ../../../datasets/COCO/coco_subset_20.txt ]]; then
      mkdir -p ../../../datasets/COCO/images
      curl -fsSL -o ../../../datasets/COCO/images/bus.jpg \
        https://raw.githubusercontent.com/airockchip/rknn_model_zoo/main/assets/bus.jpg
      echo "./images/bus.jpg" >../../../datasets/COCO/coco_subset_20.txt
    fi
  fi
  python convert.py "$ONNX_MODEL" "$PLATFORM" "$DTYPE" "$OUT_RKNN"
)

if [[ ! -f "$OUT_RKNN" ]]; then
  echo "ERROR: RKNN file not created: $OUT_RKNN" >&2
  exit 1
fi

ls -lh "$OUT_RKNN"
echo ""
echo "=== Done ==="
echo "Venv:  source $VENV_DIR/bin/activate"
echo "Cache: $WORKDIR"
echo "Copy the kit (or just the .rknn) to your Orange Pi 5, then:"
echo "  bash scripts/setup_on_orangepi.sh"
echo "  python3 -m yolo_rknn.check_npu"
