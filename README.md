# YOLO → RKNN tool (Orange Pi 5 / RK3588)

Standalone toolkit to **export a YOLO model to Rockchip RKNN** on an x86 PC and **run it on the NPU** of an Orange Pi 5 (RK3588).

This folder is self-contained. You can copy or zip it and use it without the rest of any parent repository.

## Layout

```
yolo_rknn_orangepi5/
├── README.md
├── scripts/
│   ├── export_on_pc.sh          # PC x86: .pt → .rknn (default rk3588)
│   └── setup_on_orangepi.sh     # Pi aarch64: install lite2 + librknnrt
├── yolo_rknn/                   # Python package (no ROS / no parent-repo imports)
│   ├── platform.py
│   ├── detector.py
│   └── check_npu.py
├── models/                      # Put .pt / .onnx / .rknn here
└── examples/
    └── infer_camera.py
```

## Two different Rockchip packages

| Package | Machine | Purpose |
|---------|---------|---------|
| `rknn-toolkit2` | x86 Linux PC | Convert `.onnx` → `.rknn` |
| `rknn-toolkit-lite2` | Orange Pi aarch64 | Run `.rknn` on the NPU |

They are **not** on PyPI for the conversion toolkit. Wheels come from [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2).

Python **3.13 is unsupported** (wheels are cp38–cp312). The export script uses a local Python 3.10 venv.

## 1) Export on PC (x86 Linux)

```bash
cd yolo_rknn_orangepi5
# Ensure models/best.pt exists (a sample is bundled)
bash scripts/export_on_pc.sh
# → models/best_320_rk3588.rknn
```

Useful environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `PLATFORM` | `rk3588` | Target SoC |
| `IMGSZ` | `320` | Input size |
| `DTYPE` | `fp` | `fp` (float) or `i8` (quantized) |

Manual one-liner (after installing toolkit2):

```bash
yolo export model=best.pt format=rknn name=rk3588 imgsz=320
```

## 2) Copy to Orange Pi 5

Copy the whole kit, or at least the `.rknn`:

```bash
scp -r yolo_rknn_orangepi5 ubuntu@<pi-ip>:~/
# or
scp models/best_320_rk3588.rknn ubuntu@<pi-ip>:~/yolo_rknn_orangepi5/models/
```

## 3) Install runtime on the Pi

```bash
cd ~/yolo_rknn_orangepi5
sudo bash scripts/setup_on_orangepi.sh
python3 -m yolo_rknn.check_npu
```

## 4) Run inference

```bash
python3 examples/infer_camera.py --camera /dev/video0
python3 examples/infer_camera.py --image photo.jpg --show
```

## Bundled sample weights

| File | Notes |
|------|-------|
| `models/best.pt` | YOLO person checkpoint |
| `models/best_320.onnx` | ONNX @ 320 |
| `models/best_320_rk3588.rknn` | Already converted for RK3588 |

Replace these with your own trained weights as needed.

## Requirements summary

**PC (export):** Linux x86_64, Python 3.10, git, network (clones toolkit + model zoo).

**Orange Pi 5 (infer):** aarch64, NPU driver present, Python 3.8–3.12, OpenCV + NumPy.
