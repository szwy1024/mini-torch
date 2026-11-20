"""Metal 数值后端；仅注册 float32 矩阵乘法，不处理 Value 或反向图。"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
from threading import Lock
from typing import Callable
import warnings
import weakref

from .._checks import _check_same_dtype
from ..storage import _Meta
from .base import Backend


def _build_library() -> Path:
    """首次使用时编译原生桥接，后续复用项目 tmp/metal 下的动态库。"""
    source = Path(__file__).parent / "native" / "metal_runtime.mm"
    sdk = subprocess.check_output(
        ["xcrun", "--show-sdk-path"], text=True
    ).strip()
    identity = source.read_bytes() + sdk.encode() + platform.machine().encode()
    digest = hashlib.sha256(identity).hexdigest()[:16]
    directory = Path(__file__).parents[2] / "tmp" / "metal" / digest
    library = directory / "libmini_metal.dylib"
    if not library.exists():
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "xcrun", "clang++", "-std=c++17", "-O3", "-fobjc-arc",
                "-dynamiclib", "-isysroot", sdk,
                "-isystem", str(Path(sdk) / "usr/include/c++/v1"),
                str(source), "-o", str(library),
                "-framework", "Foundation", "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
            ],
            check=True, capture_output=True, text=True,
        )
    return library


class _MetalRuntime:
    """ctypes 只传递连续缓冲区；GPU 完成后才返回 CPU 可读的结果。"""

    def __init__(self) -> None:
        self.library = ctypes.CDLL(str(_build_library()))
        self.library.mini_metal_create.argtypes = []
        self.library.mini_metal_create.restype = ctypes.c_void_p
        self.library.mini_metal_destroy.argtypes = [ctypes.c_void_p]
        self.library.mini_metal_destroy.restype = None
        self.library.mini_metal_device_name.argtypes = [ctypes.c_void_p]
        self.library.mini_metal_device_name.restype = ctypes.c_char_p
        pointer = ctypes.POINTER(ctypes.c_float)
        self.library.mini_metal_matmul.argtypes = [
            ctypes.c_void_p, pointer, pointer, pointer,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ]
        self.library.mini_metal_matmul.restype = ctypes.c_int
        self.context = self.library.mini_metal_create()
        if not self.context:
            raise RuntimeError("未找到支持 MPS 的 Metal 设备")
        self._cleanup = weakref.finalize(
            self, self.library.mini_metal_destroy, self.context
        )

    @property
    def device_name(self) -> str:
        return self.library.mini_metal_device_name(self.context).decode()

    def matmul(self, x: _Meta, y: _Meta, result: _Meta) -> None:
        left = (ctypes.c_float * x.numel).from_buffer(x.data)
        right = (ctypes.c_float * y.numel).from_buffer(y.data)
        output = (ctypes.c_float * result.numel).from_buffer(result.data)
        rows, inner = x.shape
        columns = y.shape[1]
        status = self.library.mini_metal_matmul(
            self.context, left, right, output, rows, inner, columns
        )
        if status != 0:
            raise RuntimeError(f"Metal 矩阵乘法失败，原生状态码：{status}")


class MetalBackend(Backend):
    def __init__(self, name: str = "metal") -> None:
        super().__init__(name)
        self._runtime: _MetalRuntime | None = None
        self._initialized = False
        self._lock = Lock()
        self.unavailable_reason: str | None = None

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        if sys.platform != "darwin":
            return {}
        return {"matmul": self.matmul}

    def is_available(self) -> bool:
        # 导入框架不编译、不创建设备；第一次 float32 matmul 才初始化。
        with self._lock:
            if not self._initialized:
                self._initialized = True
                if sys.platform != "darwin":
                    self.unavailable_reason = "Metal 只支持 macOS"
                else:
                    try:
                        self._runtime = _MetalRuntime()
                    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                        self.unavailable_reason = str(error)
                        warnings.warn(
                            f"Metal 不可用，将使用 CPU：{error}", RuntimeWarning,
                            stacklevel=2,
                        )
        return self._runtime is not None

    def supports(self, op: str, x: _Meta, y: _Meta | float | None) -> bool:
        if x.dtype != "float32":
            return False
        if not isinstance(y, _Meta) or y.dtype != "float32":
            return False
        return self.is_available()

    @property
    def device_name(self) -> str:
        if not self.is_available() or self._runtime is None:
            raise RuntimeError(self.unavailable_reason)
        return self._runtime.device_name

    def matmul(self, x: _Meta, y: _Meta) -> _Meta:
        _check_same_dtype(x, y)
        if x.dtype != "float32":
            raise TypeError("Metal matmul 只支持 float32")
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("matmul 当前只支持两个二维矩阵")
        rows, inner = x.shape
        if inner != y.shape[0]:
            raise ValueError(f"矩阵内维不匹配：{x.shape} @ {y.shape}")
        if not self.is_available() or self._runtime is None:
            raise RuntimeError(self.unavailable_reason)
        columns = y.shape[1]
        result = _Meta([0.0] * (rows * columns), (rows, columns), dtype=x.dtype)
        if rows and inner and columns:
            self._runtime.matmul(x, y, result)
        return result
