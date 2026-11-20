"""CPU 参考数值实现；输入输出均为 _Meta。"""

from __future__ import annotations

from math import exp, log, prod
import math
from typing import Callable, Sequence

from .._checks import (
    _check_floating_dtype, _check_indices, _check_same_dtype, _check_same_shape,
)
from ..storage import _Meta
from .base import Backend


class CPUBackend(Backend):
    """CPU 数值实现：只处理 _Meta，维度已由 Dispatcher 规范化。"""

    def __init__(self, name: str):
        super().__init__(name)

    def add(self, x: _Meta, y: _Meta) -> _Meta:
        _check_same_shape(x, y)
        _check_same_dtype(x, y)
        return _Meta(
            [a + b for a, b in zip(x.data, y.data, strict=True)], x.shape, dtype=x.dtype
        )

    def sub(self, x: _Meta, y: _Meta) -> _Meta:
        _check_same_shape(x, y)
        _check_same_dtype(x, y)
        return _Meta(
            [a - b for a, b in zip(x.data, y.data, strict=True)], x.shape, dtype=x.dtype
        )

    def mul(self, x: _Meta, y: _Meta) -> _Meta:
        """逐元素乘法，区别于 matmul。"""
        _check_same_shape(x, y)
        _check_same_dtype(x, y)
        return _Meta(
            [a * b for a, b in zip(x.data, y.data, strict=True)], x.shape, dtype=x.dtype
        )

    def div(self, x: _Meta, y: _Meta) -> _Meta:
        """同形状逐元素相除；不执行矩阵求逆或广播。"""
        _check_same_shape(x, y)
        _check_same_dtype(x, y)
        _check_floating_dtype(x.dtype)
        return _Meta(
            [a / b for a, b in zip(x.data, y.data, strict=True)], x.shape, dtype=x.dtype
        )

    def log(self, x: _Meta) -> _Meta:
        """逐元素自然对数。"""
        _check_floating_dtype(x.dtype)
        return _Meta([log(value) for value in x.data], x.shape, dtype=x.dtype)

    def scale(self, x: _Meta, scalar: float) -> _Meta:
        """矩阵或张量的数乘，scalar 是普通 Python 常数。"""
        _check_floating_dtype(x.dtype)
        return _Meta([value * scalar for value in x.data], x.shape, dtype=x.dtype)

    def matmul(self, x: _Meta, y: _Meta) -> _Meta:
        """(m, k) @ (k, n) → (m, n)，仅支持二维矩阵。"""
        _check_same_dtype(x, y)
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise ValueError("matmul 当前只支持两个二维矩阵")
        m, k = x.shape
        other_k, n = y.shape
        if k != other_k:
            raise ValueError(f"矩阵内维不匹配：{x.shape} @ {y.shape}")
        x_data, y_data = x.data, y.data
        result: list[float] = [0] * (m * n)
        for row in range(m):
            row_start = row * k
            for col in range(n):
                # 行优先：x[row, inner] 和 y[inner, col] 的一维下标。
                result[row * n + col] = sum(
                    x_data[row_start + inner] * y_data[inner * n + col]
                    for inner in range(k)
                )
        return _Meta(result, (m, n), dtype=x.dtype)

    def clone(self, x: _Meta) -> _Meta:
        return x.clone()

    def transpose(self, x: _Meta) -> _Meta:
        """二维转置，返回独立的数据副本，不创建共享存储的视图。"""
        if len(x.shape) != 2:
            raise ValueError("transpose 当前只支持二维矩阵")
        rows, cols = x.shape
        return _Meta(
            [x.data[row * cols + col] for col in range(cols) for row in range(rows)],
            (cols, rows),
            dtype=x.dtype,
        )

    def sum(self, x: _Meta, dim: int | None = None, *, keepdim: bool = False) -> _Meta:
        """沿单个维度求和；dim=None 求全部元素之和，空集合的和为 0。"""
        if dim is None:
            if keepdim:
                shape = (1,) * x.ndim
            else:
                shape = ()
            return _Meta([sum(x.data)], shape, dtype=x.dtype)

        # 标量允许 dim=0/-1；归约后仍为标量。
        if x.ndim == 0:
            return x.clone()
        outer = prod(x.shape[:dim])
        reduce_size = x.shape[dim]
        inner = prod(x.shape[dim + 1 :])
        result: list[float] = []
        for outer_index in range(outer):
            for inner_index in range(inner):
                total: float = 0
                for reduce_index in range(reduce_size):
                    offset = (
                        outer_index * reduce_size + reduce_index
                    ) * inner + inner_index
                    total += x.data[offset]
                result.append(total)
        if keepdim:
            middle = (1,)
        else:
            middle = ()
        shape = x.shape[:dim] + middle + x.shape[dim + 1 :]
        return _Meta(result, shape, dtype=x.dtype)

    def expand(self, x: _Meta, dim: int, size: int) -> _Meta:
        """本例定义：在 dim 插入长度为 size 的新维，并复制数据。

        等价于 torch.unsqueeze(x, dim) 后将新维 expand 到 size，再物化副本。
        例如 (2,) 在 dim=1,size=3 时变成 (2,3)，并非 PyTorch 的 expand 接口。
        """
        outer = prod(x.shape[:dim])
        inner = prod(x.shape[dim:])
        result: list[float] = []
        for outer_index in range(outer):
            block = x.data[outer_index * inner : (outer_index + 1) * inner]
            for _ in range(size):
                result.extend(block)
        return _Meta(result, x.shape[:dim] + (size,) + x.shape[dim:], dtype=x.dtype)

    def reshape(self, x: _Meta, shape: Sequence[int]) -> _Meta:
        """连续数据的形状转换，本例复制数据，暂不实现共享存储的 view。"""
        target = tuple(shape)
        if prod(target) != x.numel:
            raise ValueError(f"reshape 元素数不匹配：{x.shape} → {target}")
        return _Meta(x.data, target, dtype=x.dtype)

    def exponential(self, x: _Meta) -> _Meta:
        _check_floating_dtype(x.dtype)
        return _Meta([exp(data) for data in x.data], x.shape, dtype=x.dtype)

    def relu(self, x: _Meta) -> _Meta:
        return _Meta([max(data, 0) for data in x.data], x.shape, dtype=x.dtype)

    def sqrt(self, x: _Meta) -> _Meta:
        _check_floating_dtype(x.dtype)
        return _Meta([math.sqrt(v) for v in x.data], x.shape, dtype=x.dtype)

    def max(self, x: _Meta, dim: int | None = None, *, keepdim: bool = False) -> _Meta:
        return self._max_with_mask(x, dim, keepdim)[0]

    def max_mask(
        self, x: _Meta, dim: int | None = None, *, keepdim: bool = False
    ) -> _Meta:
        """返回输入形状的首个最大位置掩码"""
        return self._max_with_mask(x, dim, keepdim)[1]

    def _max_with_mask(
        self, x: _Meta, dim: int | None, keepdim: bool
    ) -> tuple[_Meta, _Meta]:
        """返回最大值和首个最大位置掩码；并列最大值选择第一个。"""
        if dim is None or x.ndim == 0:
            winner = max(range(x.numel), key=x.data.__getitem__)
            best = x.data[winner]
            mask = [0] * x.numel
            mask[winner] = 1
            if keepdim:
                shape = (1,) * x.ndim
            else:
                shape = ()
            return _Meta([best], shape, dtype=x.dtype), _Meta(mask, x.shape, dtype=x.dtype)
        axis = dim
        width = x.shape[axis]
        outer, inner = prod(x.shape[:axis]), prod(x.shape[axis + 1 :])
        output, mask = [], [0] * x.numel
        for before in range(outer):
            for after in range(inner):
                offsets = [(before * width + j) * inner + after for j in range(width)]
                winner = max(offsets, key=lambda index: x.data[index])
                output.append(x.data[winner])
                mask[winner] = 1
        if keepdim:
            middle = (1,)
        else:
            middle = ()
        shape = x.shape[:axis] + middle + x.shape[axis + 1 :]
        return _Meta(output, shape, dtype=x.dtype), _Meta(mask, x.shape, dtype=x.dtype)

    def embedding(self, tokens: _Meta, weight: _Meta) -> _Meta:
        _check_indices(tokens)
        _check_floating_dtype(weight.dtype)
        embed_dim = weight.shape[-1]
        result: list[float] = []

        for token_id in tokens.data:
            index = int(token_id)
            start = index * embed_dim
            result.extend(weight.data[start : start + embed_dim])

        output_shape = tokens.shape + (embed_dim,)
        return _Meta(result, output_shape, dtype=weight.dtype)

    def slice(self, x: _Meta, indices: _Meta, dim: int) -> _Meta:
        """沿 dim 轴按一维索引选取切片，保留索引的顺序和重复项。"""
        _check_indices(indices)
        outer = prod(x.shape[:dim])
        inner = prod(x.shape[dim + 1 :])
        length = x.shape[dim]
        result: list[float] = []
        for before in range(outer):
            for index in indices.data:
                start = (before * length + int(index)) * inner
                result.extend(x.data[start : start + inner])
        shape = x.shape[:dim] + (indices.numel,) + x.shape[dim + 1 :]
        return _Meta(result, shape, dtype=x.dtype)

    def scatter(self, x: _Meta, indices: _Meta, dim: int, size: int) -> _Meta:
        """把切片累加回指定位置；未选位置为零，重复索引累加。"""
        _check_indices(indices)
        outer = prod(x.shape[:dim])
        inner = prod(x.shape[dim + 1 :])
        length = x.shape[dim]
        shape = x.shape[:dim] + (size,) + x.shape[dim + 1 :]
        result: list[float] = [0] * prod(shape)
        for before in range(outer):
            for position, index in enumerate(indices.data):
                source = (before * length + position) * inner
                target = (before * size + int(index)) * inner
                for after in range(inner):
                    result[target + after] += x.data[source + after]
        return _Meta(result, shape, dtype=x.dtype)

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        return {
            "add": self.add,
            "sub": self.sub,
            "mul": self.mul,
            "div": self.div,
            "log": self.log,
            "scale": self.scale,
            "matmul": self.matmul,
            "transpose": self.transpose,
            "clone": self.clone,
            "sum": self.sum,
            "expand": self.expand,
            "reshape": self.reshape,
            "exponential": self.exponential,
            "relu": self.relu,
            "max": self.max,
            "max_mask": self.max_mask,
            "sqrt": self.sqrt,
            "embedding": self.embedding,
            "slice": self.slice,
            "scatter": self.scatter,
        }

