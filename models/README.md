# Models

Place weights in this directory:

| File | Role |
|------|------|
| `best.pt` | YOLO PyTorch checkpoint (input for export) |
| `best_320.onnx` | Optional ONNX intermediate (cached by export) |
| `best_320_rk3588.rknn` | RKNN model for **Orange Pi 5 / RK3588** |

Naming convention: `best_<imgsz>_<platform>.rknn`

- Orange Pi 5 → `rk3588`
- Orange Pi 3B → `rk3566`

RK3566 and RK3588 models are **not** interchangeable.
