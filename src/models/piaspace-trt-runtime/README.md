# piaspace-trt-runtime

Shared TensorRT runner + engine provisioning helpers consumed by
`piaspace-clip-reid` and `piaspace-yolo26`.

This package owns the bits the model packages agree on:

- `TRTRunner` — torch.cuda-based TRT engine runner (no `pycuda`).
- `ModelSpec` — dataclass describing one provisionable artifact.
- `download_onnx` / `build_engine` / `ensure_engine` — HF→ONNX→engine
  pipeline; the per-family `MODELS` registry is supplied by the caller.

Self-contained: depends only on external packages.

## Usage

```python
from piaspace_trt_runtime import ModelSpec, TRTRunner, ensure_engine

MODELS: dict[str, ModelSpec] = {
    "my_model": ModelSpec(
        onnx_filename="my_model.onnx",
        engine_filename="my_model.fp16.engine",
        input_name="image",
        min_shape=(1, 3, 224, 224),
        opt_shape=(4, 3, 224, 224),
        max_shape=(16, 3, 224, 224),
        hub_repo="my-org/my-repo",
        hub_path="onnx/my_model.onnx",
    ),
}

engine_path = ensure_engine("my_model", MODELS, weights_dir="./weights")
runner = TRTRunner(engine_path, device="cuda:0")
features = runner.infer_single(image_tensor)
```
