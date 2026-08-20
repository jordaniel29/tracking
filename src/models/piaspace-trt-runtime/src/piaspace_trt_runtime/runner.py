"""Thin TensorRT engine runner (torch.cuda-based, no pycuda).

``tensorrt`` is lazy-imported, so a torch-only deployment (no engines) imports
this module fine.

    runner = TRTRunner("weights/some_engine.fp16.engine", device="cuda:0")
    feats = runner.infer_single(x_fp32_NCHW)   # torch.Tensor on the same device

The runner does no normalization/preprocessing — callers feed already-
preprocessed tensors and consume raw network outputs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _import_trt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "tensorrt is not installed. Install a TRT build matching the engine "
            "you intend to load (e.g. `pip install tensorrt==10.5.0`)."
        ) from e
    return trt


_TRT_LOGGER = None


def _get_logger() -> Any:
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        trt = _import_trt()
        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return _TRT_LOGGER


_TRT_TO_TORCH_DTYPE: dict[Any, torch.dtype] = {}


def _trt_to_torch_dtype(trt_dtype: Any) -> torch.dtype:
    if not _TRT_TO_TORCH_DTYPE:
        trt = _import_trt()
        _TRT_TO_TORCH_DTYPE.update(
            {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT8: torch.int8,
                trt.DataType.INT32: torch.int32,
                trt.DataType.BOOL: torch.bool,
            }
        )
        for attr, torch_t in (
            ("BF16", torch.bfloat16),
            ("UINT8", torch.uint8),
            ("INT64", torch.int64),
        ):
            if hasattr(trt.DataType, attr):
                _TRT_TO_TORCH_DTYPE[getattr(trt.DataType, attr)] = torch_t
    return _TRT_TO_TORCH_DTYPE[trt_dtype]


class TRTRunner:
    """Loads a serialized TRT engine and runs inference with torch.cuda buffers."""

    def __init__(self, engine_path: str | Path, device: str = "cuda:0") -> None:
        trt = _import_trt()
        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")
        self.engine_path = engine_path
        self.device = torch.device(device)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        # Pin the device for the whole construction: TRT binds to the "current"
        # CUDA context at deserialize time, so building on cuda:1 while torch
        # defaults to cuda:0 otherwise yields "invalid resource handle".
        with torch.cuda.device(self.device):
            self._runtime = trt.Runtime(_get_logger())
            self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
            if self._engine is None:
                raise RuntimeError(
                    f"Failed to deserialize engine at {engine_path}. Usually a TRT "
                    "version mismatch — engines are not portable across TRT minor versions."
                )
            self._context = self._engine.create_execution_context()

            self.input_names: list[str] = []
            self.output_names: list[str] = []
            self._dtypes: dict[str, torch.dtype] = {}
            self._is_input: dict[str, bool] = {}
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                is_in = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                self._is_input[name] = is_in
                self._dtypes[name] = _trt_to_torch_dtype(self._engine.get_tensor_dtype(name))
                (self.input_names if is_in else self.output_names).append(name)

            self._stream = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
        self._out_buffers: dict[tuple[Any, ...], torch.Tensor] = {}

        logger.info(
            "TRTRunner loaded %s on %s | inputs=%s outputs=%s",
            engine_path.name,
            self.device,
            self.input_names,
            self.output_names,
        )

    @property
    def num_inputs(self) -> int:
        return len(self.input_names)

    @property
    def num_outputs(self) -> int:
        return len(self.output_names)

    def get_binding_shape(self, name: str) -> tuple[int, ...]:
        """Static binding shape from the engine. Dynamic dims appear as -1."""
        return tuple(self._engine.get_tensor_shape(name))

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run one inference pass. Output buffers are reused across calls — clone
        if you need to keep an output past the next ``infer``."""
        missing = set(self.input_names) - set(inputs.keys())
        if missing:
            raise ValueError(f"missing inputs: {missing}")

        with torch.cuda.device(self.device):
            inputs = dict(inputs)
            for name, tensor in list(inputs.items()):
                if not self._is_input.get(name, False):
                    raise ValueError(f"'{name}' is not an input of this engine")
                if not tensor.is_cuda or tensor.device != self.device:
                    tensor = tensor.to(self.device)
                tensor = tensor.contiguous()
                self._context.set_input_shape(name, tuple(tensor.shape))
                self._context.set_tensor_address(name, tensor.data_ptr())
                inputs[name] = tensor  # keep reference alive past the loop

            outputs: dict[str, torch.Tensor] = {}
            for name in self.output_names:
                shape = tuple(self._context.get_tensor_shape(name))
                dtype = self._dtypes[name]
                key = (name, shape, dtype)
                buf = self._out_buffers.get(key)
                if buf is None:
                    buf = torch.empty(shape, dtype=dtype, device=self.device)
                    self._out_buffers[key] = buf
                self._context.set_tensor_address(name, buf.data_ptr())
                outputs[name] = buf

            ok = self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
            if not ok:
                raise RuntimeError("TRT execute_async_v3 returned False")
            self._stream.synchronize()
        return outputs

    def infer_single(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience for single-input / single-output engines (the common case)."""
        if self.num_inputs != 1 or self.num_outputs != 1:
            raise RuntimeError(
                f"infer_single requires 1-in/1-out; have "
                f"{self.num_inputs}-in/{self.num_outputs}-out"
            )
        out = self.infer({self.input_names[0]: x})
        return out[self.output_names[0]]
