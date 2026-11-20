"""算子配置、处理层选择与数值后端派发。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ._checks import _normalize_dim
from .autograd_layer import Autograd
from .backends import CPUBackend, GPUBackend
from .storage import _Meta
from .value import Value


@dataclass(frozen=True)
class OpOptions:
    """在处理层之间传递不可变的算子配置；配置不参与梯度计算。"""

    dim: int | None = None  # 已规范化的非负下标；None 表示归约所有维度。
    size: int | None = None
    keepdim: bool = False
    shape: tuple[int, ...] | None = None



class ExecuteLayer:
    def __init__(self, dispatcher: Dispatch) -> None:
        self.dispatcher = dispatcher
        cpu = CPUBackend("cpu")
        gpu = GPUBackend("gpu")
        self.kernels: dict[str, dict[str, Callable[..., _Meta]]] = {
            "cpu": cpu.get_ability(),
            "gpu": gpu.get_ability(),
        }

    def apply(
        self,
        op: str,
        x: Value,
        y: Value | float | None,
        options: OpOptions,
        excluded: frozenset[str],
    ) -> Value:
        if isinstance(y, Value):
            argument = y.meta
        else:
            argument = y
        result = self.call_kernel(op, x.meta, argument, options=options)
        return Value(result, name=op, require_grad=False)

    def call_kernel(
        self, op: str, x: _Meta, y: _Meta | float | None = None, *, options: OpOptions
    ) -> _Meta:
        # 仍固定选 CPU；GPUBackend 只是占位。
        kernel = self.kernels["cpu"][op]
        if self.dispatcher._trace:
            print(f"  CPU kernel: {op}")
        if op in ("sum", "max", "max_mask"):
            return kernel(x, options.dim, keepdim=options.keepdim)
        if op in ("sqrt", "transpose", "clone", "exponential", "relu", "log"):
            return kernel(x)
        if op == "expand":
            return kernel(x, options.dim, options.size)
        if op == "reshape":
            return kernel(x, options.shape)
        if op == "slice":
            return kernel(x, y, options.dim)
        if op == "scatter":
            return kernel(x, y, options.dim, options.size)
        return kernel(x, y)

    def get_kernels(self) -> dict[str, dict[str, Callable[..., _Meta]]]:
        return self.kernels



class Dispatch:
    """配置只能通过关键字传入。"""

    def __init__(self, *, _trace: bool = False) -> None:
        self._trace = _trace
        self.autograd = Autograd(self)
        self.executeLayer = ExecuteLayer(self)
        self.kernels = self.executeLayer.get_kernels()
        self.layers: dict[
            str,
            Callable[
                [str, Value, Value | float | None, OpOptions, frozenset[str]], Value
            ],
        ] = {
            "Autograd": self.autograd.apply,
            "CPU": self.executeLayer.apply,
        }
        self.priority = ("Autograd", "CPU")

    def call(
        self,
        op: str,
        x: Value,
        y: Value | float | None = None,
        *,
        dim: int | None = None,
        size: int | None = None,
        keepdim: bool = False,
        shape: Sequence[int] | None = None,
    ) -> Value:
        if op == "scale" and isinstance(y, (int, float)):
            y = float(y)
        if dim is not None:
            dim = _normalize_dim(dim, x.meta.ndim, insert=op == "expand")
        if shape is not None:
            target = tuple(shape)
        else:
            target = None
        options = OpOptions(dim, size, keepdim, target)
        return self.redispatch(op, x, y, options=options, exclude=frozenset())

    def redispatch(
        self,
        op: str,
        x: Value,
        y: Value | float | None = None,
        *,
        options: OpOptions,
        exclude: frozenset[str],
    ) -> Value:
        for key in self.priority:
            if key not in exclude:
                if self._trace:
                    print(f"Dispatcher: {op} → {key}；排除 {sorted(exclude)}")
                return self.layers[key](op, x, y, options, exclude)
        raise RuntimeError(f"{op} 没有可用处理层，已排除 {sorted(exclude)}")



_dispatch = Dispatch()
