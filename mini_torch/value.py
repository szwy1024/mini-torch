"""用户操作张量的接口，以及可学习参数标记。"""

from __future__ import annotations

from array import array
from typing import Sequence

from ._checks import _check_floating_dtype
from .autograd import LeafNode, Node, backward
from .storage import DType, _Meta


class Value:
    def __init__(
        self,
        meta: _Meta,
        name: str | None = None,
        require_grad: bool = True,
        node: Node | None = None,
    ) -> None:
        if require_grad:
            _check_floating_dtype(meta.dtype)
        self.meta = meta
        self.name = name
        self.grad: Value | None = None
        self.require_grad = require_grad
        self.node = node
        if require_grad and node is None:
            self.node = LeafNode(self)

    @property
    def data(self) -> array[float] | array[int]:
        return self.meta.data

    @property
    def dtype(self) -> DType:
        return self.meta.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.meta.shape

    def __add__(self, other: Value) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("add", self, other)

    def __sub__(self, other: Value) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("sub", self, other)

    def __mul__(self, other: Value | float | int) -> Value:
        from .dispatcher import _dispatch

        if isinstance(other, (int, float)):
            return _dispatch.call("scale", self, float(other))
        return _dispatch.call("mul", self, other)

    def __truediv__(self, other: Value) -> Value:
        """当前只支持两个同形状的 Value 相除。"""
        from .dispatcher import _dispatch

        return _dispatch.call("div", self, other)

    def __rmul__(self, scalar: float | int) -> Value:
        return self * scalar

    def __matmul__(self, other: Value) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("matmul", self, other)

    def clone(self) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("clone", self)

    def transpose(self) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("transpose", self)

    # 可以直接使用.T进行矩阵转置运算
    @property
    def T(self) -> Value:
        return self.transpose()

    def sum(self, dim: int | None = None, *, keepdim: bool = False) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("sum", self, dim=dim, keepdim=keepdim)

    def expand(self, dim: int, size: int) -> Value:
        """在 dim 插入并复制新维度。"""
        from .dispatcher import _dispatch

        return _dispatch.call("expand", self, dim=dim, size=size)

    def reshape(self, shape: Sequence[int]) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("reshape", self, shape=shape)

    def exp(self) -> Value:
        return self.exponential()

    def log(self) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("log", self)

    def exponential(self) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("exponential", self)

    def relu(self) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("relu", self)

    def sqrt(self) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("sqrt", self)

    def max(self, dim: int | None = None, *, keepdim: bool = False) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("max", self, dim=dim, keepdim=keepdim)

    def broadcast(self, shape: Sequence[int]) -> Value:
        """从右对齐各维，长度为 1 的维度可以扩展。"""
        target = tuple(shape)
        if len(target) < len(self.shape):
            raise ValueError("广播后的维数不能少于输入")
        aligned = (1,) * (len(target) - len(self.shape)) + self.shape
        result = self.reshape(aligned)
        for axis, (source, size) in enumerate(zip(aligned, target, strict=True)):
            if source == size:
                continue
            if source != 1:
                raise ValueError(f"不能将 {self.shape} 广播到 {target}")
            reduced = result.shape[:axis] + result.shape[axis + 1 :]
            result = result.reshape(reduced).expand(axis, size)
        return result

    def mean(self, dim: int | None = None, *, keepdim: bool = False) -> Value:
        result = self.sum(dim, keepdim=keepdim)
        if dim is None:
            count = self.meta.numel
        else:
            if self.shape:
                count = self.shape[dim]
            else:
                # shape=() 表示标量，只有一个元素
                count = 1
        return result * (1.0 / count)

    def softmax(self, dim: int = -1) -> Value:
        maximum = self.max(dim, keepdim=True)
        shifted = self - maximum.broadcast(self.shape)
        numerator = shifted.exp()
        return numerator / numerator.sum(dim, keepdim=True).broadcast(self.shape)

    def log_softmax(self, dim: int = -1) -> Value:
        maximum = self.max(dim, keepdim=True)
        shifted = self - maximum.broadcast(self.shape)
        normalizer = shifted.exp().sum(dim, keepdim=True).log()
        return shifted - normalizer.broadcast(self.shape)

    def embedding(self, weight: Value) -> Value:
        from .dispatcher import _dispatch

        return _dispatch.call("embedding", self, weight)

    def slice(self, indices: Value, dim: int = -1) -> Value:
        """沿 dim 按索引选取切片。"""
        from .dispatcher import _dispatch

        return _dispatch.call("slice", self, indices, dim=dim)

    def scatter(self, indices: Value, size: int, dim: int = -1) -> Value:
        """将切片累加到长度为 size 的轴上。"""
        from .dispatcher import _dispatch

        return _dispatch.call("scatter", self, indices, size=size, dim=dim)

    def __repr__(self) -> str:
        return f"Value(data={self.data.tolist()}, shape={self.shape}, dtype={self.dtype}, require_grad={self.require_grad})"

    # create_graph 表示是否在反向计算过程中再次构造反向计算图
    def backward(
        self,
        seed: Value | None = None,
        *,
        create_graph: bool = False,
        _trace: bool = False,
    ) -> None:
        backward(self, seed, create_graph=create_graph, _trace=_trace)



class Parameter(Value):
    """标记：这是模型参数，而不是普通的中间计算结果。"""

    pass

