"""保留的 GPU 后端占位；当前实现仍在 CPU 执行。"""

from __future__ import annotations

from typing import Callable

from .._checks import _check_same_dtype, _check_same_shape
from ..storage import _Meta
from .base import Backend


class GPUBackend(Backend):
    def __init__(self, name: str):
        super().__init__(name)

    def add(self, x: _Meta, y: _Meta) -> _Meta:
        _check_same_shape(x, y)
        _check_same_dtype(x, y)
        return _Meta(
            [a + b for a, b in zip(x.data, y.data, strict=True)], x.shape, dtype=x.dtype
        )

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        return {"add": self.add}

