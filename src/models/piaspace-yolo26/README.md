# piaspace-yolo26

PIASPACE YOLO26 object detector wrapper. Image in → list of `Detection`
records (bbox + class + conf + logical class-group name) out. Built on
`ultralytics.YOLO`; TRT engine auto-provisioning via `piaspace-trt-runtime`.

Self-contained: depends only on external packages plus `piaspace-trt-runtime`.

## Usage

```python
from piaspace_yolo26 import YOLO26Detector, Detection

det = YOLO26Detector(cfg["detector"])
detections = det.detect_image(frame_bgr)  # list[Detection]
```

Class-group routing — the YAML `target_classes` maps logical group names
to lists of COCO class ids, so the downstream pipeline can route crops
to the right ReID model:

```yaml
detector:
  model: yolo26l.fp16.engine   # auto-provisioned from PIA-SPACE-LAB/SSAVE
  #     yolo26n.fp16.engine    # nano variant — also auto-provisioned (smaller/faster)
  #     <name>.pt              # any .pt runs native PyTorch (no TRT build)
  weights_dir: ./weights
  device: cuda:0
  conf: 0.35
  imgsz: 640
  target_classes:
    person:  [0]
    vehicle: [2, 5, 7]   # car, bus, truck
```

In this repository the detector is consumed by `pia_tracking.pipeline`
(see `config/tracking.yaml` at the repository root for the config actually
used).
