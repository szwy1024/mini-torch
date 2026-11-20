"""连续 CPU 存储及张量形状、数据类型；不包含自动微分信息。"""

from __future__ import annotations

from array import array
from math import prod
from typing import Literal, Sequence

DType = Literal["float64", "float32", "int32"]


class Storage:
    """连续的 CPU 数值缓冲区；dtype 与数组的实际元素格式一致。"""

    def __init__(self, data: Sequence[float], dtype: DType = "float64") -> None:
        self.dtype: DType = dtype
        self.data: array[float] | array[int]
        if dtype == "int32":
            integers = []
            for value in data:
                if not isinstance(value, int):
                    raise TypeError("int32 数据必须是整数")
                integers.append(value)
            self.data = array("i", integers)
        elif dtype == "float32":
            self.data = array("f", data)
        elif dtype == "float64":
            self.data = array("d", data)
        else:
            raise TypeError(f"不支持 dtype={dtype}")



class _Meta:
    """数值张量描述：存储 + 形状；不包含自动微分信息。"""

    def __init__(
        self, data: Sequence[float], shape: Sequence[int], *, dtype: DType = "float64"
    ) -> None:
        # 固定形状，避免外部修改传入的 shape 列表影响已有张量。
        self.shape = tuple(shape)
        if any(dim < 0 for dim in self.shape):
            raise ValueError("shape 不能包含负数")
        self.storage = Storage(data, dtype)
        if len(self.data) != self.numel:
            raise ValueError("数据长度与 shape 不匹配")

    @property
    def data(self) -> array[float] | array[int]:
        return self.storage.data

    @property
    def dtype(self) -> DType:
        return self.storage.dtype

    @property
    def numel(self) -> int:
        # shape=() 表示标量，需要一个元素；shape=(0, 3) 则需要零个元素。
        return prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def clone(self) -> _Meta:
        return _Meta(self.data, self.shape, dtype=self.dtype)

    @staticmethod
    def zeros_like(meta: _Meta) -> _Meta:
        return _Meta([0] * meta.numel, meta.shape, dtype=meta.dtype)

    @staticmethod
    def ones_like(meta: _Meta) -> _Meta:
        return _Meta([1] * meta.numel, meta.shape, dtype=meta.dtype)

    def __repr__(self) -> str:
        return f"_Meta(data={self.data.tolist()}, shape={self.shape}, dtype={self.dtype})"

