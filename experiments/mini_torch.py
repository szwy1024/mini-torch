from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from math import exp, log
import math
import random
from itertools import product as cartesian_product
from typing import Any, Callable, Generator, Generic, Iterator, Sequence, TypeVar

# ContextVar 保存当前执行上下文的状态，嵌套或异常退出时用 token 恢复。
_grad_enabled: ContextVar[bool] = ContextVar("grad_enabled", default=True)


def is_grad_enabled() -> bool:
    return _grad_enabled.get()


@contextmanager
def set_grad_enabled(enabled: bool) -> Generator[None, None, None]:
    """本例只支持 with 用法；不修改任何输入的 require_grad。"""
    token = _grad_enabled.set(enabled)
    try:
        yield
    finally:
        # 退出时恢复上下文
        _grad_enabled.reset(token)


def no_grad() -> AbstractContextManager[None]:
    return set_grad_enabled(False)


def enable_grad() -> AbstractContextManager[None]:
    return set_grad_enabled(True)


class Storage:
    """实际数值。教学简化：用 list[float]，而不是底层字节缓冲区。"""

    def __init__(self, data: Sequence[float]) -> None:
        self.data = [float(value) for value in data]


class _Meta:
    """数值张量描述：存储 + 形状；不包含自动微分信息。"""

    def __init__(self, data: Sequence[float], shape: Sequence[int]) -> None:
        # 固定形状，避免外部修改传入的 shape 列表影响已有张量。
        self.shape = tuple(shape)
        if any(type(dim) is not int or dim < 0 for dim in self.shape):
            raise ValueError("shape 的各维长度必须是非负整数")
        self.storage = Storage(data)
        if len(self.data) != self.numel:
            raise ValueError(
                f"数据长度 {len(self.data)} 与 shape={self.shape} "
                f"需要的元素数 {self.numel} 不一致"
            )

    @property
    def data(self) -> list[float]:
        return self.storage.data

    @property
    def numel(self) -> int:
        # shape=() 表示标量，需要一个元素；shape=(0, 3) 则需要零个元素。
        return prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def clone(self) -> _Meta:
        return _Meta(self.data, self.shape)

    @staticmethod
    def zeros_like(meta: _Meta) -> _Meta:
        return _Meta([0.0] * meta.numel, meta.shape)

    @staticmethod
    def ones_like(meta: _Meta) -> _Meta:
        return _Meta([1.0] * meta.numel, meta.shape)

    def __repr__(self) -> str:
        return f"_Meta(data={self.data}, shape={self.shape})"


def _normalize_dim(dim: int, ndim: int, *, insert: bool = False) -> int:
    """普通维度范围 [-ndim, ndim-1]；插入位置范围 [-ndim-1, ndim]。"""
    if type(dim) is not int:
        raise TypeError("dim 必须是整数，不能是 bool 或浮点数")
    limit = ndim + 1 if insert else max(ndim, 1)
    if not -limit <= dim < limit:
        raise ValueError(f"dim={dim} 超出范围；输入维数={ndim}，insert={insert}")
    return dim + limit if dim < 0 else dim


def _check_size(size: int) -> None:
    if type(size) is not int:
        raise TypeError("size 必须是整数，不能是 bool 或浮点数")
    if size < 0:
        raise ValueError("size 必须大于或等于 0")


def _check_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(shape)
    if any(type(dim) is not int or dim < 0 for dim in result):
        raise ValueError("shape 必须由非负整数组成")
    return result


def _check_div_inputs(x: _Meta, y: _Meta) -> None:
    """当前除法不做广播；正零和负零都不允许作为分母。"""
    if x.shape != y.shape:
        raise ValueError(f"需要相同形状，实际为 {x.shape} 和 {y.shape}")
    if any(value == 0.0 for value in y.data):
        raise ZeroDivisionError("div 的分母不能包含零")


def _check_log_input(x: _Meta) -> None:
    """只接受实数正值；写成 not > 0 也会拒绝 NaN。"""
    if any(not value > 0.0 for value in x.data):
        raise ValueError("log 的输入必须逐元素大于零")


@dataclass(frozen=True)
class OpOptions:
    """在处理层之间传递不可变的算子配置；配置不参与梯度计算。"""

    dim: int | None = None
    size: int | None = None
    keepdim: bool = False
    shape: tuple[int, ...] | None = None
    axes: tuple[int, ...] | None = None


class Backend:
    def __init__(self, name: str):
        self.name = name

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        raise NotImplementedError


class CPUBackend(Backend):
    """CPU 数值实现：只处理 _Meta，不创建反向节点。"""

    @staticmethod
    def _check_same_shape(x: _Meta, y: _Meta) -> None:
        if x.shape != y.shape:
            raise ValueError(f"需要相同形状，实际为 {x.shape} 和 {y.shape}")

    def __init__(self, name: str):
        super().__init__(name)

    def add(self, x: _Meta, y: _Meta) -> _Meta:
        self._check_same_shape(x, y)
        return _Meta([a + b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def sub(self, x: _Meta, y: _Meta) -> _Meta:
        self._check_same_shape(x, y)
        return _Meta([a - b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def mul(self, x: _Meta, y: _Meta) -> _Meta:
        """逐元素乘法，区别于 matmul。"""
        self._check_same_shape(x, y)
        return _Meta([a * b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def div(self, x: _Meta, y: _Meta) -> _Meta:
        """同形状逐元素相除；不执行矩阵求逆或广播。"""
        _check_div_inputs(x, y)
        return _Meta([a / b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def log(self, x: _Meta) -> _Meta:
        """逐元素自然对数。"""
        _check_log_input(x)
        return _Meta([log(value) for value in x.data], x.shape)

    def scale(self, x: _Meta, scalar: float) -> _Meta:
        """矩阵或张量的数乘，scalar 是普通 Python 常数。"""
        return _Meta([value * scalar for value in x.data], x.shape)

    def matmul(self, x: _Meta, y: _Meta) -> _Meta:
        """最后两维矩阵乘法；Value 层已显式对齐 batch 维。"""
        if x.ndim < 2 or y.ndim < 2 or x.shape[:-2] != y.shape[:-2]:
            raise ValueError("matmul 需要至少二维且 batch 维相同的输入")
        rows, inner = x.shape[-2:]
        other_inner, cols = y.shape[-2:]
        if inner != other_inner:
            raise ValueError("matmul 矩阵内维不匹配")
        batch = prod(x.shape[:-2])
        result = []
        for group in range(batch):
            for row in range(rows):
                for col in range(cols):
                    result.append(sum(x.data[group*rows*inner + row*inner+k]
                                      * y.data[group*inner*cols+k*cols+col]
                                      for k in range(inner)))
        return _Meta(result, x.shape[:-2] + (rows, cols))

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
        )

    def sum(self, x: _Meta, dim: int | None = None, *, keepdim: bool = False) -> _Meta:
        """沿单个维度求和；dim=None 求全部元素之和，空集合的和为 0。"""
        if type(keepdim) is not bool:
            raise TypeError("keepdim 必须是 bool")
        if dim is None:
            shape = (1,) * x.ndim if keepdim else ()
            return _Meta([sum(x.data)], shape)

        dim = _normalize_dim(dim, x.ndim)
        # 标量允许 dim=0/-1；归约后仍为标量。
        if x.ndim == 0:
            return x.clone()
        outer = prod(x.shape[:dim])
        reduce_size = x.shape[dim]
        inner = prod(x.shape[dim + 1 :])
        result: list[float] = []
        for outer_index in range(outer):
            for inner_index in range(inner):
                total = 0.0
                for reduce_index in range(reduce_size):
                    offset = (
                        outer_index * reduce_size * inner
                        + reduce_index * inner
                        + inner_index
                    )
                    total += x.data[offset]
                result.append(total)
        middle = (1,) if keepdim else ()
        shape = x.shape[:dim] + middle + x.shape[dim + 1 :]
        return _Meta(result, shape)

    def expand(self, x: _Meta, dim: int, size: int) -> _Meta:
        """本例定义：在 dim 插入长度为 size 的新维，并复制数据。

        等价于 torch.unsqueeze(x, dim) 后将新维 expand 到 size，再物化副本。
        例如 (2,) 在 dim=1,size=3 时变成 (2,3)，并非 PyTorch 的 expand 接口。
        """
        dim = _normalize_dim(dim, x.ndim, insert=True)
        _check_size(size)
        outer = prod(x.shape[:dim])
        inner = prod(x.shape[dim:])
        result: list[float] = []
        for outer_index in range(outer):
            block = x.data[outer_index * inner : (outer_index + 1) * inner]
            for _ in range(size):
                result.extend(block)
        return _Meta(result, x.shape[:dim] + (size,) + x.shape[dim:])

    def reshape(self, x: _Meta, shape: Sequence[int]) -> _Meta:
        """连续数据的形状转换，本例复制数据，暂不实现共享存储的 view。"""
        target = _check_shape(shape)
        if prod(target) != x.numel:
            raise ValueError(f"reshape 元素数不匹配：{x.shape} → {target}")
        return _Meta(x.data, target)

    def exponential(self, x: _Meta) -> _Meta:
        return _Meta([exp(data) for data in x.data], x.shape)

    def relu(self, x: _Meta) -> _Meta:
        return _Meta([max(data, 0) for data in x.data], x.shape)

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        return {
            "permute": self.permute,
            "max": self.max,
            "sqrt": self.sqrt,
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
        }

    def sqrt(self, x: _Meta) -> _Meta:
        _check_sqrt_input(x)
        return _Meta([math.sqrt(v) for v in x.data], x.shape)

    def max(self, x: _Meta, dim: int | None = None, *, keepdim: bool = False) -> _Meta:
        return _max_reduce(x, dim, keepdim)[0]

    def permute(self, x: _Meta, axes: Sequence[int]) -> _Meta:
        axes = _check_permutation(axes, x.ndim)
        shape = tuple(x.shape[axis] for axis in axes)
        inverse = tuple(axes.index(axis) for axis in range(x.ndim))
        data = []
        for index in cartesian_product(*(range(width) for width in shape)):
            source = tuple(index[inverse[axis]] for axis in range(x.ndim))
            data.append(x.data[_flat_offset(source, x.shape)])
        return _Meta(data, shape)


class GPUBackend(Backend):
    @staticmethod
    def _check_same_shape(x: _Meta, y: _Meta) -> None:
        if x.shape != y.shape:
            raise ValueError(f"需要相同形状，实际为 {x.shape} 和 {y.shape}")

    def __init__(self, name: str):
        super().__init__(name)

    def add(self, x: _Meta, y: _Meta) -> _Meta:
        self._check_same_shape(x, y)
        return _Meta([a + b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        return {"add": self.add}


class Node:
    """反向规则的输入、输出现在都是 Value，可以继续参与自动微分。"""

    def __init__(self, edge: list[Node | None], name: str) -> None:
        self.edge = edge
        self.name = name

    def apply(self, grad_out: Value) -> tuple[Value, ...]:
        raise NotImplementedError("具体节点必须实现自己的反向行为")


class LeafNode(Node):
    """累加叶子梯度；累加和复制也在 Engine 指定的梯度模式下执行。"""

    def __init__(self, value: Value) -> None:
        if not value.require_grad:
            raise ValueError("不需要梯度的 Value 不应创建 LeafNode")
        super().__init__(edge=[], name=f"LeafNode({value.name})")
        self.value = value

    def apply(self, grad_out: Value) -> tuple[Value, ...]:
        if grad_out.shape != self.value.shape:
            raise ValueError("叶子的梯度形状必须与数据相同")
        previous = self.value.grad
        self.value.grad = grad_out.clone() if previous is None else previous + grad_out
        return ()


class InnerNode(Node):
    def __init__(
        self,
        edge: list[Node | None],
        backrule: Callable[[Value], tuple[Value, ...]],
        data: dict[str, Value] | None = None,
        name: str = "InnerNode",
    ) -> None:
        super().__init__(edge=edge, name=name)
        self.backrule = backrule
        self.data = {} if data is None else data

    def apply(self, grad_out: Value) -> tuple[Value, ...]:
        return self.backrule(grad_out)


class Value:
    def __init__(
        self,
        meta: _Meta,
        name: str | None = None,
        require_grad: bool = True,
        node: Node | None = None,
    ) -> None:
        if node is not None and not require_grad:
            raise ValueError("关闭梯度的 Value 不应关联反向节点")
        self.meta = meta
        self.name = name
        self.grad: Value | None = None
        self.require_grad = require_grad
        self.node = node
        if require_grad and node is None:
            self.node = LeafNode(self)

    @property
    def storage(self) -> Storage:
        return self.meta.storage

    @property
    def data(self) -> list[float]:
        return self.meta.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.meta.shape

    def __add__(self, other: Value) -> Value:
        return _dispatch.call("add", self, other)

    def __sub__(self, other: Value) -> Value:
        return _dispatch.call("sub", self, other)

    def __mul__(self, other: Value | float | int) -> Value:
        if isinstance(other, (int, float)):
            return _dispatch.call("scale", self, float(other))
        return _dispatch.call("mul", self, other)

    def __truediv__(self, other: Value) -> Value:
        """当前只支持两个同形状的 Value 相除。"""
        return _dispatch.call("div", self, other)

    def __rmul__(self, scalar: float | int) -> Value:
        return self * scalar

    def __matmul__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            raise TypeError("matmul 需要两个 Value")
        if self.meta.ndim < 2 or other.meta.ndim < 2:
            raise ValueError("matmul 当前不支持一维向量，输入至少二维")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError("matmul 矩阵内维不匹配")
        batch = _broadcast_shape(self.shape[:-2], other.shape[:-2])
        left_shape, right_shape = batch + self.shape[-2:], batch + other.shape[-2:]
        left = self if self.shape == left_shape else self.broadcast_to(left_shape)
        right = other if other.shape == right_shape else other.broadcast_to(right_shape)
        return _dispatch.call("matmul", left, right)

    def clone(self) -> Value:
        return _dispatch.call("clone", self)

    def transpose(self) -> Value:
        return _dispatch.call("transpose", self)

    def sum(self, dim: int | None = None, *, keepdim: bool = False) -> Value:
        return _dispatch.call("sum", self, dim=dim, keepdim=keepdim)

    def expand(self, dim: int, size: int) -> Value:
        """在 dim 插入并复制新维度，详见 CPUBackend.expand。"""
        return _dispatch.call("expand", self, dim=dim, size=size)

    def reshape(self, shape: Sequence[int]) -> Value:
        return _dispatch.call("reshape", self, shape=shape)

    def broadcast_to(self, shape: Sequence[int]) -> Value:
        """广播到目标形状；组合已有 Value 运算，不新增 kernel 或反向规则。

        例如 (3,) → (2,3)，或 (2,1) → (2,3)。从右侧对齐维度，
        每个输入维度必须与目标相等，或长度为 1；1 也可以扩展成 0。
        返回物化副本，不共享存储；本接口暂不接受 -1 推断。
        """
        target = _check_shape(shape)
        if len(target) < len(self.shape):
            raise ValueError(
                f"broadcast_to 目标维数不能少于输入：{self.shape} → {target}"
            )

        # 缺少的前导维度视为 1；先完整验证，避免非法目标触发部分计算和建图。
        aligned = (1,) * (len(target) - len(self.shape)) + self.shape
        for axis, (source_size, target_size) in enumerate(zip(aligned, target, strict=True)):
            if source_size != target_size and source_size != 1:
                raise ValueError(
                    f"broadcast_to 形状不兼容：{self.shape} → {target}；"
                    f"对齐后的第 {axis} 维 {source_size} 不能广播为 {target_size}"
                )

        # reshape 同时补齐前导维度并返回独立 Value，保持 no_grad 行为一致。
        result = self.reshape(aligned)
        for axis, (source_size, target_size) in enumerate(zip(aligned, target, strict=True)):
            if source_size == target_size:
                continue
            # 现有 expand 只能插入新轴：先移除该长度为 1 的轴，再插入复制。
            reduced = result.shape[:axis] + result.shape[axis + 1:]
            result = result.reshape(reduced).expand(axis, target_size)
        return result

    def exp(self) -> Value:
        return self.exponential()

    def log(self) -> Value:
        return _dispatch.call("log", self)

    def exponential(self) -> Value:
        return _dispatch.call("exponential", self)

    def relu(self) -> Value:
        return _dispatch.call("relu", self)

    # 可以直接使用.T进行矩阵转置运算
    @property
    def T(self) -> Value:
        return self.transpose()

    # create_graph 表示是否在反向计算过程中再次构造反向计算图
    def backward(
        self,
        seed: Value | None = None,
        *,
        create_graph: bool = False,
        _trace: bool = False,
    ) -> None:
        backward(self, seed, create_graph=create_graph, _trace=_trace)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, shape={self.shape}, require_grad={self.require_grad})"

    def mean(self, dim: int | None = None, *, keepdim: bool = False) -> Value:
        axis = None if dim is None else _normalize_dim(dim, self.meta.ndim)
        count = self.meta.numel if axis is None else (self.shape[axis] if self.shape else 1)
        if count == 0:
            raise ValueError("当前教学版不对长度为零的轴求平均")
        return self.sum(dim, keepdim=keepdim) * (1.0 / count)

    def sqrt(self) -> Value:
        return _dispatch.call("sqrt", self)

    def max(self, dim: int | None = None, *, keepdim: bool = False) -> Value:
        return _dispatch.call("max", self, dim=dim, keepdim=keepdim)

    def softmax(self, dim: int = -1) -> Value:
        axis = _check_softmax(self, dim)
        shifted = self - self.max(axis, keepdim=True).broadcast_to(self.shape)
        numerator = shifted.exp()
        return numerator / numerator.sum(axis, keepdim=True).broadcast_to(self.shape)

    def log_softmax(self, dim: int = -1) -> Value:
        axis = _check_softmax(self, dim)
        shifted = self - self.max(axis, keepdim=True).broadcast_to(self.shape)
        normalizer = shifted.exp().sum(axis, keepdim=True).log()
        return shifted - normalizer.broadcast_to(self.shape)

    def permute(self, axes: Sequence[int]) -> Value:
        return _dispatch.call("permute", self, axes=axes)

    def swapaxes(self, first: int, second: int) -> Value:
        first, second = _normalize_dim(first, self.meta.ndim), _normalize_dim(second, self.meta.ndim)
        axes = list(range(self.meta.ndim))
        axes[first], axes[second] = axes[second], axes[first]
        return self.permute(axes)


class Autograd:
    """前向和反向的 Value 运算共用此层，是否建图由当前梯度模式决定。"""

    def __init__(self, dispatcher: Dispatch) -> None:
        self.dispatcher = dispatcher

    # excluded表示需要排除的处理流程
    def apply(
        self,
        op: str,
        x: Value,
        y: Value | float | None,
        options: OpOptions,
        excluded: frozenset[str],
    ) -> Value:
        # 只有value类型可能需要梯度，纯数值不关联梯度信息
        other_requires_grad = y.require_grad if isinstance(y, Value) else False
        # 查询当前上下文是否开启了梯度记录。
        require_grad = is_grad_enabled() and (x.require_grad or other_requires_grad)
        node: Node | None = None
        if self.dispatcher._trace:
            print(
                f"  Autograd: grad_enabled={is_grad_enabled()}, record={require_grad}"
            )

        if require_grad:
            if op == "clone":
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out,),
                    name=f"CloneBackward({x.name})",
                )
            elif op == "transpose":
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out.T,),
                    name=f"TransposeBackward({x.name})",
                )
            elif op == "exponential":
                saved = {"save": x.clone()}
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out * saved["save"].exp(),),
                    name=f"ExpBackward({x.name})",
                )
            elif op == "permute":
                axes = options.axes
                if axes is None:
                    raise RuntimeError("permute 缺少已验证的维度排列")
                inverse = tuple(axes.index(axis) for axis in range(len(axes)))
                node = InnerNode([x.node], lambda g: (g.permute(inverse),), name="PermuteBackward")
            elif op == "max":
                _, mask_meta = _max_reduce(x.meta, options.dim, options.keepdim)
                mask = Value(mask_meta, require_grad=False)
                if options.dim is None or x.meta.ndim == 0:
                    expanded_shape = (1,) * x.meta.ndim
                else:
                    axis = options.dim
                    expanded_shape = x.shape[:axis] + (1,) + x.shape[axis+1:]
                def max_backward(g: Value) -> tuple[Value, ...]:
                    return (g.reshape(expanded_shape).broadcast_to(x.shape) * mask,)
                node = InnerNode([x.node], max_backward, data={"mask": mask}, name="MaxBackward")
            elif op == "sqrt":
                saved = {"input": x.clone()}
                node = InnerNode([x.node],
                    lambda g: (g / (saved["input"].sqrt() * 2.0),),
                    data=saved, name=f"SqrtBackward({x.name})")
            elif op == "log":
                saved = {"input": x.clone()}
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out / saved["input"],),
                    data=saved,
                    name=f"LogBackward({x.name})",
                )
            elif op == "relu":
                mask = Value(
                    _Meta([1.0 if value > 0 else 0.0 for value in x.data], x.shape),
                    name=f"ReluMask({x.name})",
                    require_grad=False,
                )
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out * mask,),
                    data={"mask": mask},
                    name=f"ReluBackward({x.name})",
                )
            elif op == "scale":
                if not isinstance(y, float):
                    raise TypeError("scale 的第二个参数必须是 float")
                scalar = y
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out * scalar,),
                    name=f"ScaleBackward({x.name}, {scalar})",
                )
            elif op == "sum":
                input_shape = x.shape
                axis = options.dim
                # 如果axis为空，计算结果是标量
                if axis is None:

                    def sum_all_backward(grad_out: Value) -> tuple[Value, ...]:
                        result = grad_out.reshape(())
                        for insert_axis, length in enumerate(input_shape):
                            result = result.expand(insert_axis, length)
                        return (result,)

                    node = InnerNode(
                        [x.node], sum_all_backward, name=f"SumBackward({x.name})"
                    )
                # 如果x本身就是标量
                elif x.meta.ndim == 0:
                    node = InnerNode(
                        [x.node],
                        lambda grad_out: (grad_out,),
                        name=f"SumBackward({x.name})",
                    )
                else:
                    sum_axis = axis
                    reduced_shape = input_shape[:sum_axis] + input_shape[sum_axis + 1 :]
                    reduce_size = input_shape[sum_axis]

                    def sum_axis_backward(grad_out: Value) -> tuple[Value, ...]:
                        # keepdim=True 时先去掉长度为 1 的归约轴，再插入复制。
                        gradient = grad_out.reshape(reduced_shape)
                        return (gradient.expand(sum_axis, reduce_size),)

                    node = InnerNode(
                        [x.node], sum_axis_backward, name=f"SumBackward({x.name})"
                    )
            elif op == "expand":
                expand_axis = options.dim
                if expand_axis is None:
                    raise RuntimeError("expand 缺少已验证的 dim")
                # 前向插入复制的是新轴，反向将其求和并删除，恢复原形状。
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out.sum(expand_axis),),
                    name=f"ExpandBackward({x.name})",
                )
            elif op == "reshape":
                original_shape = x.shape
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out.reshape(original_shape),),
                    name=f"ReshapeBackward({x.name})",
                )

            else:
                if not isinstance(y, Value):
                    raise TypeError(f"{op} 的第二个参数必须是 Value")
                edges = [x.node, y.node]
                if op == "add":
                    node = InnerNode(
                        edges,
                        lambda grad_out: (grad_out, grad_out),
                        name=f"AddBackward({x.name}, {y.name})",
                    )
                elif op == "sub":
                    node = InnerNode(
                        edges,
                        lambda grad_out: (grad_out, grad_out * -1.0),
                        name=f"SubBackward({x.name}, {y.name})",
                    )
                elif op == "div":
                    saved = {"left": x.clone(), "right": y.clone()}

                    def div_backward(grad_out: Value) -> tuple[Value, ...]:
                        left = saved["left"]
                        right = saved["right"]
                        grad_left = grad_out / right
                        # 等价于 -grad_out * left / right²。
                        # 分两次除以 right，避免先平方让很小的分母下溢为零。
                        grad_right = (grad_out * -1.0) * (left / right) / right
                        return (grad_left, grad_right)

                    node = InnerNode(
                        edges,
                        div_backward,
                        data=saved,
                        name=f"DivBackward({x.name}, {y.name})",
                    )
                elif op in ("mul", "matmul"):
                    # 保留前向快照，同时保留它到原输入的求导关系。
                    # x.meta.clone() 只有数值；x.clone() 则通过 CloneBackward 连接 x。
                    saved = {"left": x.clone(), "right": y.clone()}
                    if op == "mul":

                        def mul_backward(grad_out: Value) -> tuple[Value, ...]:
                            return (grad_out * saved["right"], grad_out * saved["left"])

                        backrule = mul_backward
                    else:

                        def matmul_backward(grad_out: Value) -> tuple[Value, ...]:
                            return (
                                grad_out @ saved["right"].swapaxes(-1, -2),
                                saved["left"].swapaxes(-1, -2) @ grad_out,
                            )

                        backrule = matmul_backward
                    node = InnerNode(
                        edges,
                        backrule,
                        data=saved,
                        name=f"{op.capitalize()}Backward({x.name}, {y.name})",
                    )
                else:
                    raise NotImplementedError(f"缺少 {op} 的反向规则")

        # 排除 Autograd 只影响本次算子的后续派发，不改变当前梯度模式。
        result = self.dispatcher.redispatch(
            op, x, y, options=options, exclude=excluded | {"Autograd"}
        )
        return Value(result.meta, result.name, require_grad, node)


class ExecuteLayer:
    def __init__(self, dispatcher: Dispatch) -> None:
        self.dispatcher = dispatcher
        cpu = CPUBackend("cpu")
        gpu = GPUBackend("gpu")
        self.kernels: dict[str, dict[str, Callable[..., _Meta]]] = {
            "cpu": cpu.get_ability(),
            "gpu": gpu.get_ability(),
        }

    def apply(self, op: str, x: Value, y: Value | float | None,
              options: OpOptions, excluded: frozenset[str]) -> Value:
        argument = y.meta if isinstance(y, Value) else y
        result = self.call_kernel(op, x.meta, argument, options=options)
        return Value(result, name=op, require_grad=False)

    def call_kernel(
        self, op: str, x: _Meta, y: _Meta | float | None = None, *, options: OpOptions
    ) -> _Meta:
        # 仍固定选 CPU；GPUBackend 只是占位，未实现 Metal 或设备存储。
        kernel = self.kernels["cpu"][op]
        if self.dispatcher._trace:
            print(f"  CPU kernel: {op}")
        if op == "permute":
            return kernel(x, options.axes)
        if op == "max":
            return kernel(x, options.dim, keepdim=options.keepdim)
        if op == "sqrt":
            return kernel(x)
        if op in ("transpose", "clone", "exponential", "relu", "log"):
            return kernel(x)
        if op == "sum":
            return kernel(x, options.dim, keepdim=options.keepdim)
        if op == "expand":
            return kernel(x, options.dim, options.size)
        if op == "reshape":
            return kernel(x, options.shape)
        return kernel(x, y)

    def get_kernels(self) -> dict[str, dict[str, Callable[..., _Meta]]]:
        return self.kernels


class Dispatch:
    """配置只能通过关键字传入。"""

    def __init__(self, *, _trace: bool = False) -> None:
        self._trace = _trace
        self.operator = frozenset(
            {
                "sqrt",
                "max",
                "permute",
                "add",
                "sub",
                "mul",
                "div",
                "log",
                "scale",
                "matmul",
                "transpose",
                "clone",
                "sum",
                "expand",
                "reshape",
                "exponential",
                "relu",
            }
        )
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
        axes: Sequence[int] | None = None,
    ) -> Value:
        # 此层只整理配置；数值实现负责检查形状和数学定义域。
        if op == "scale" and isinstance(y, (int, float)):
            y = float(y)
        if dim is not None:
            dim = _normalize_dim(dim, x.meta.ndim, insert=op == "expand")
        target = tuple(shape) if shape is not None else None
        permutation = _check_permutation(axes, x.meta.ndim) if op == "permute" else None
        options = OpOptions(dim, size, keepdim, target, permutation)
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
        if op not in self.operator:
            raise NotImplementedError(f"不支持 Value 算子 {op}")
        for key in self.priority:
            if key not in exclude:
                if self._trace:
                    print(f"Dispatcher: {op} → {key}；排除 {sorted(exclude)}")
                return self.layers[key](op, x, y, options, exclude)
        raise RuntimeError(f"{op} 没有可用处理层，已排除 {sorted(exclude)}")


_dispatch = Dispatch()


class AutogradEngine:
    """梯度收齐后统一调用 Node.apply；梯度缓冲区属于本次反向任务。"""

    def run_backward(
        self,
        output: Node,
        seed: Value,
        *,
        create_graph: bool = False,
        _trace: bool = False,
    ) -> None:
        # 默认反向临时关闭建图；create_graph=True 则临时开启。
        # 无论调用者外面的梯度模式是什么，退出后都会恢复。
        with set_grad_enabled(create_graph):
            nodes = self._collect_nodes(output)
            pending = self._compute_dependencies(nodes)
            grad_buffers: dict[Node, Value] = {output: seed}
            ready: deque[Node] = deque([output])
            if _trace:
                print(f"\n=== Engine：create_graph={create_graph} ===")
                for node in nodes:
                    print(f"{node.name}: 等待 {pending[node]} 份梯度")
            while ready:
                node = ready.popleft()
                grad_out = grad_buffers.pop(node)
                self._evaluate_function(
                    node, grad_out, pending, grad_buffers, ready, _trace
                )

    def _collect_nodes(self, output: Node) -> list[Node]:
        nodes: list[Node] = []
        seen: set[Node] = set()
        stack: list[Node] = [output]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.append(node)
            stack.extend(edge for edge in node.edge if edge is not None)
        return nodes

    def _compute_dependencies(self, nodes: list[Node]) -> dict[Node, int]:
        pending = {node: 0 for node in nodes}
        for node in nodes:
            for edge in node.edge:
                if edge is not None:
                    pending[edge] += 1
        return pending

    def _evaluate_function(
        self,
        node: Node,
        grad_out: Value,
        pending: dict[Node, int],
        grad_buffers: dict[Node, Value],
        ready: deque[Node],
        _trace: bool,
    ) -> None:
        if _trace:
            print(f"\n执行 {node.name}：上游梯度 {grad_out}")
        contributions = node.apply(grad_out)
        for slot, (edge, contribution) in enumerate(
            zip(node.edge, contributions, strict=True), start=1
        ):
            if edge is None:
                continue
            previous = grad_buffers.get(edge)
            grad_buffers[edge] = (
                contribution if previous is None else previous + contribution
            )
            pending[edge] -= 1
            if _trace:
                print(
                    f"  输入边 {slot} → {edge.name}："
                    f"累计 {grad_buffers[edge]}；还等 {pending[edge]} 份"
                )
            if pending[edge] == 0:
                ready.append(edge)


_engine = AutogradEngine()


class SGD:
    """只更新不同的叶子参数；保持 Parameter 对象和存储身份不变。"""
    def __init__(self, params: Iterator[Parameter], lr: float = 0.01) -> None:
        self.params = list(dict.fromkeys(params))
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.data[:] = (param - param.grad * self.lr).data


Optime = SGD  # 兼容旧阶段的教学命名。


def backward(
    output: Value,
    seed: Value | None = None,
    *,
    create_graph: bool = False,
    _trace: bool = False,
) -> None:
    if not output.require_grad or output.node is None:
        raise RuntimeError("当前 Value 没有可用的反向入口")
    if seed is None:
        if output.meta.numel != 1:
            raise ValueError("多元素输出必须显式提供同形状的上游梯度 seed")
        seed = Value(_Meta.ones_like(output.meta), name="seed", require_grad=False)
    elif not isinstance(seed, Value):
        raise TypeError("seed 必须是 Value；单元素输出可以省略 seed 或传入 None")
    if seed.shape != output.shape:
        raise ValueError(f"seed.shape={seed.shape} 与输出形状 {output.shape} 不一致")
    _engine.run_backward(output.node, seed, create_graph=create_graph, _trace=_trace)


class Parameter(Value):
    """标记：这是模型参数，而不是普通的中间计算结果。"""

    pass


# 不同模块接收不同类型的输入：Linear 接收 Value，Embedding 接收 token 序列。
InputT = TypeVar("InputT", contravariant=True)
TokenInput = Sequence[int] | Sequence[Sequence[int]]


class Module(Generic[InputT]):
    _parameters: dict[str, Parameter]
    _modules: dict[str, Module[Any]]  # 注册表只遍历子模块，不调用它们的 forward。

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self,"_modules",{})
        self.name = name

    def __setattr__(self, name: str, value: object) -> None:
        parameter = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if isinstance(value, Parameter):
            if parameter is None:
                raise RuntimeError("请先调用super().__init__()")
            parameter[name] = value
        elif parameter is not None:
            parameter.pop(name, None)
        if isinstance(value,Module):
            if modules is None:
                raise RuntimeError("请先调用super().__init__()")
            modules[name] = value
        elif modules is not None:
            modules.pop(name, None)
       
        object.__setattr__(self, name, value)

    def parameter(self) -> Iterator[Parameter]:
        seen_params: set[int] = set()
        seen_modules: set[int] = set()
        queue: deque[Module[Any]] = deque([self])
        while queue:
            module = queue.popleft()
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            for param in module._parameters.values():
                if id(param) not in seen_params:
                    seen_params.add(id(param))
                    yield param
            queue.extend(module._modules.values())


    def forward(self, input: InputT, /) -> Value:
        raise NotImplementedError("子模块需要实现 forward")

    def __call__(self, input: InputT, /) -> Value:
        return self.forward(input)

    def parameters(self) -> Iterator[Parameter]:
        return self.parameter()

    def __delattr__(self, name: str) -> None:
        object.__delattr__(self, name)
        self._parameters.pop(name, None)
        self._modules.pop(name, None)


class Linear(Module[Value]):
    """任意正整数输入/输出维度；末维做投影，显式广播偏置。"""
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True,
                 rng: random.Random | None = None) -> None:
        super().__init__("Linear")
        if min(in_features, out_features) < 1:
            raise ValueError("Linear 的输入、输出维度必须是正整数")
        self.in_features, self.out_features = in_features, out_features
        generator = rng if rng is not None else random
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(_Meta(
            [generator.uniform(-bound, bound) for _ in range(in_features * out_features)],
            (in_features, out_features)), name="weight")
        self.bias = Parameter(_Meta([0] * out_features, (out_features,)), name="bias") if bias else None

    def forward(self, x: Value) -> Value:
        if len(x.shape) < 2 or x.shape[-1] != self.in_features:
            raise ValueError("Linear 输入至少二维，末维必须等于 in_features")
        result = x @ self.weight
        return result if self.bias is None else result + self.bias.broadcast_to(result.shape)


def _check_sqrt_input(x: _Meta) -> None:
    if any(not value >= 0 for value in x.data):
        raise ValueError("sqrt 的实数输入必须大于或等于零")


class LayerNorm(Module[Value]):
    def __init__(self, features: int, eps: float = 1e-5) -> None:
        super().__init__("LayerNorm")
        if features <= 0 or eps <= 0:
            raise ValueError("features 和 eps 必须大于零")
        self.features, self.eps = features, float(eps)
        self.weight = Parameter(_Meta([1] * features, (features,)), name="gamma")
        self.bias = Parameter(_Meta([0] * features, (features,)), name="beta")

    def forward(self, x: Value) -> Value:
        if not x.shape or x.shape[-1] != self.features:
            raise ValueError("LayerNorm 的输入末维不匹配")
        centered = x - x.mean(-1, keepdim=True).broadcast_to(x.shape)
        variance = (centered * centered).mean(-1, keepdim=True)
        epsilon = Value(_Meta([self.eps] * variance.meta.numel, variance.shape), require_grad=False)
        normalized = centered / (variance + epsilon).sqrt().broadcast_to(x.shape)
        return normalized * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)


def _max_reduce(x: _Meta, dim: int | None, keepdim: bool) -> tuple[_Meta, _Meta]:
    """返回最大值和首个最大位置掩码；并列最大值选择第一个。"""
    if type(keepdim) is not bool:
        raise TypeError("keepdim 必须是 bool")
    if dim is not None:
        dim = _normalize_dim(dim, x.ndim)
    if any(math.isnan(v) for v in x.data):
        raise ValueError("当前 max 不接受 NaN")
    if dim is None or x.ndim == 0:
        if not x.data:
            raise ValueError("不能对空集合求 max")
        best = max(x.data)
        mask = [0.0] * x.numel
        mask[x.data.index(best)] = 1.0
        shape = (1,) * x.ndim if keepdim else ()
        return _Meta([best], shape), _Meta(mask, x.shape)
    axis = _normalize_dim(dim, x.ndim)
    width = x.shape[axis]
    if width == 0:
        raise ValueError("max 的归约轴不能为空")
    outer, inner = prod(x.shape[:axis]), prod(x.shape[axis+1:])
    output, mask = [], [0.0] * x.numel
    for before in range(outer):
        for after in range(inner):
            offsets = [(before * width + j) * inner + after for j in range(width)]
            winner = max(offsets, key=lambda index: x.data[index])
            output.append(x.data[winner])
            mask[winner] = 1.0
    shape = x.shape[:axis] + ((1,) if keepdim else ()) + x.shape[axis+1:]
    return _Meta(output, shape), _Meta(mask, x.shape)


def _check_softmax(x: Value, dim: int) -> int:
    if not x.shape:
        raise ValueError("softmax 至少需要一维输入")
    axis = _normalize_dim(dim, x.meta.ndim)
    if any(math.isnan(v) or v == math.inf for v in x.data):
        raise ValueError("softmax 不接受 NaN 或正无穷")
    row_max, _ = _max_reduce(x.meta, axis, True)
    if any(v == -math.inf for v in row_max.data):
        raise ValueError("softmax 的每组输入必须至少有一个有限值，不能全部被屏蔽")
    return axis


def causal_mask(length: int) -> Value:
    if length <= 0:
        raise ValueError("因果掩码长度必须是正整数")
    return Value(_Meta([0.0 if col <= row else -math.inf
                        for row in range(length) for col in range(length)],
                       (length, length)), name="causal_mask", require_grad=False)


class CausalSelfAttention(Module[Value]):
    def __init__(self, d_model: int, *, num_heads: int = 1, rng: random.Random | None = None) -> None:
        super().__init__("CausalSelfAttention")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model 和 num_heads 必须为正整数，且 d_model 能被 num_heads 整除")
        self.query = Linear(d_model, d_model, rng=rng)
        self.key = Linear(d_model, d_model, rng=rng)
        self.value = Linear(d_model, d_model, rng=rng)
        self.projection = Linear(d_model, d_model, rng=rng)
        self.d_model, self.num_heads = d_model, num_heads

    def forward(self, x: Value) -> Value:
        if len(x.shape) not in (2,3) or x.shape[-1] != self.d_model or any(n == 0 for n in x.shape):
            raise ValueError("Attention 输入必须是非空 (T,D) 或 (B,T,D)")
        unbatched = len(x.shape) == 2
        source = x.reshape((1,) + x.shape) if unbatched else x
        batch, length, _ = source.shape
        heads, head_dim = self.num_heads, self.d_model // self.num_heads
        def split(projection: Linear) -> Value:
            return projection(source).reshape((batch,length,heads,head_dim)).permute((0,2,1,3))
        q, k, v = split(self.query), split(self.key), split(self.value)
        scores = (q @ k.swapaxes(-1,-2)) * (head_dim ** -0.5)
        masked = scores + causal_mask(length).broadcast_to(scores.shape)
        self.attention_weights = masked.softmax(-1)
        context = (self.attention_weights @ v).permute((0,2,1,3)).reshape((batch,length,self.d_model))
        result = self.projection(context)
        return result.reshape((length,self.d_model)) if unbatched else result


class FeedForward(Module[Value]):
    def __init__(self, d_model: int, d_ff: int, *, rng: random.Random | None = None) -> None:
        super().__init__("FeedForward")
        self.up = Linear(d_model, d_ff, rng=rng)
        self.down = Linear(d_ff, d_model, rng=rng)

    def forward(self, x: Value) -> Value:
        return self.down(self.up(x).relu())


class TransformerBlock(Module[Value]):
    """Pre-LN：残差支路形状不变，FFN 末层不使用 ReLU。"""
    def __init__(self, d_model: int, d_ff: int, *, num_heads: int = 1,
                 rng: random.Random | None = None) -> None:
        super().__init__("TransformerBlock")
        self.norm1, self.norm2 = LayerNorm(d_model), LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads=num_heads, rng=rng)
        self.ffn = FeedForward(d_model, d_ff, rng=rng)

    def forward(self, x: Value) -> Value:
        h = x + self.attention(self.norm1(x))
        return h + self.ffn(self.norm2(h))


def _check_tokens(tokens: Sequence[int], vocab_size: int) -> list[int]:
    result = list(tokens)
    if not result or any(type(token) is not int or not 0 <= token < vocab_size for token in result):
        raise ValueError("token 序列非空，每个 token 必须是词表范围内的整数")
    return result


class Embedding(Module[TokenInput]):
    """用 one-hot @ weight 展示嵌入查表的可微等价实现，暂不优化索引开销。"""
    def __init__(self, vocab_size: int, d_model: int, *, rng: random.Random | None = None) -> None:
        super().__init__("Embedding")
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("词表和嵌入维度必须是正整数")
        generator = rng if rng is not None else random
        self.vocab_size, self.d_model = vocab_size, d_model
        self.weight = Parameter(_Meta([generator.uniform(-.5,.5) for _ in range(vocab_size*d_model)],
                                     (vocab_size,d_model)), name="embedding")

    def forward(self, tokens: TokenInput) -> Value:
        flat, shape = _token_layout(tokens, self.vocab_size)
        one_hot = [float(token == col) for token in flat for col in range(self.vocab_size)]
        return Value(_Meta(one_hot, shape + (self.vocab_size,)), require_grad=False) @ self.weight


class PositionalEncoding(Module[Value]):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__("PositionalEncoding")
        if d_model <= 0 or max_seq_len <= 0:
            raise ValueError("位置编码维度和最大长度必须是正整数")
        self.d_model, self.max_seq_len = d_model, max_seq_len

    def forward(self, x: Value) -> Value:
        if len(x.shape) not in (2,3) or x.shape[-1] != self.d_model or not 0 < x.shape[-2] <= self.max_seq_len:
            raise ValueError("位置编码输入形状或长度不匹配")
        length = x.shape[-2]
        values = [(math.sin if j%2 == 0 else math.cos)(pos / (10000 ** (2*(j//2)/self.d_model)))
                  for pos in range(length) for j in range(self.d_model)]
        positions = Value(_Meta(values, (length,self.d_model)), require_grad=False)
        return x + positions.broadcast_to(x.shape)


def cross_entropy(logits: Value, targets: TokenInput) -> Value:
    if len(logits.shape) not in (2,3) or any(n == 0 for n in logits.shape):
        raise ValueError("交叉熵需要非空 (T,V) 或 (B,T,V) logits")
    if not all(math.isfinite(v) for v in logits.data):
        raise ValueError("交叉熵 logits 必须是有限值")
    labels, shape = _token_layout(targets, logits.shape[-1])
    if shape != logits.shape[:-1]:
        raise ValueError("标签形状与 logits 的 batch/序列维不匹配")
    one_hot = Value(_Meta([float(label == col) for label in labels for col in range(logits.shape[-1])],
                          logits.shape), require_grad=False)
    return (logits.log_softmax(-1) * one_hot).sum() * (-1.0 / len(labels))


class TinyTransformerLM(Module[TokenInput]):
    def __init__(self, vocab_size: int, d_model: int = 4, d_ff: int = 8,
                 max_seq_len: int = 16, num_layers: int = 1, seed: int = 42,
                 num_heads: int = 1) -> None:
        super().__init__("TinyTransformerLM")
        if num_layers <= 0:
            raise ValueError("num_layers 必须是正整数")
        rng = random.Random(seed)
        self.embedding = Embedding(vocab_size, d_model, rng=rng)
        self.position = PositionalEncoding(d_model, max_seq_len)
        self.block_names = tuple(f"block_{i}" for i in range(num_layers))
        for name in self.block_names:
            setattr(self, name, TransformerBlock(d_model, d_ff, num_heads=num_heads, rng=rng))
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng=rng)
        self.max_seq_len, self.vocab_size = max_seq_len, vocab_size

    def forward(self, tokens: TokenInput) -> Value:
        x = self.position(self.embedding(tokens))
        for name in self.block_names:
            x = getattr(self, name)(x)
        return self.head(self.norm(x))

    def generate(self, context: Sequence[int], steps: int) -> list[int]:
        result = _check_tokens(context, self.vocab_size)
        if steps < 0:
            raise ValueError("生成步数必须是非负整数")
        with no_grad():
            for _ in range(steps):
                scores = self(result[-self.max_seq_len:]).data[-self.vocab_size:]
                result.append(max(range(self.vocab_size), key=lambda i: scores[i]))
        return result


def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    rank = max(len(left), len(right))
    a, b = (1,) * (rank-len(left)) + left, (1,) * (rank-len(right)) + right
    result = []
    for x, y in zip(a, b, strict=True):
        if x == y:
            result.append(x)
        elif x == 1:
            result.append(y)
        elif y == 1:
            result.append(x)
        else:
            raise ValueError(f"batch 维度不兼容：{left} 与 {right}")
    return tuple(result)


def _check_permutation(axes: Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axes is None:
        raise TypeError("permute 需要 axes")
    values = tuple(axes)
    if len(values) != ndim:
        raise ValueError("permute 必须指定每一个维度")
    normalized = tuple(_normalize_dim(axis, ndim) for axis in values)
    if len(set(normalized)) != ndim:
        raise ValueError("permute 的维度不能重复")
    return normalized


def _flat_offset(index: Sequence[int], shape: tuple[int, ...]) -> int:
    offset = 0
    for i, width in zip(index, shape, strict=True):
        offset = offset * width + i
    return offset


def _token_layout(tokens: TokenInput, vocab_size: int) -> tuple[list[int], tuple[int, ...]]:
    rows = list(tokens)
    if not rows:
        raise ValueError("token 输入不能为空")
    flat = [token for token in rows if isinstance(token, int)]
    if len(flat) == len(rows):
        return _check_tokens(flat, vocab_size), (len(rows),)
    sequences = [row for row in rows if isinstance(row, Sequence)]
    if len(sequences) != len(rows):
        raise ValueError("token 必须是整数序列或等长的二维整数序列")
    checked = [_check_tokens(row, vocab_size) for row in sequences]
    width = len(checked[0])
    if any(len(row) != width for row in checked):
        raise ValueError("batch 中的序列长度必须相同；当前不做 padding")
    return [token for row in checked for token in row], (len(rows), width)


def train_demo(steps: int = 80, seed: int = 42, num_heads: int = 2) -> tuple[TinyTransformerLM, list[float]]:
    model=TinyTransformerLM(3,d_model=6,d_ff=12,num_heads=num_heads,max_seq_len=6,seed=seed)
    tokens=[[0,1,2,0,1,2],[1,2,0,1,2,0],[2,0,1,2,0,1]]
    targets=[[1,2,0,1,2,0],[2,0,1,2,0,1],[0,1,2,0,1,2]]
    optimizer=SGD(model.parameters(),lr=.06)
    history=[]
    for step in range(steps):
        optimizer.zero_grad()
        loss=cross_entropy(model(tokens),targets)
        history.append(loss.data[0])
        loss.backward()
        optimizer.step()
        if step == 0 or (step+1)%20 == 0 or step == steps-1:
            print(f"step={step:03d} cross_entropy={loss.data[0]:.6f}")
    print("greedy tokens:",model.generate([0,1,2],9))
    return model,history


def regression_demo(steps: int = 80, seed: int = 7) -> None:
    rng=random.Random(seed)
    first,last=Linear(2,8,rng=rng),Linear(8,1,rng=rng)
    x=Value(_Meta([-1,-1,-1,1,1,-1,1,1],(4,2)),require_grad=False)
    target=Value(_Meta([1.5,-4.5,5.5,-.5],(4,1)),require_grad=False)
    optimizer=SGD(iter([*first.parameters(),*last.parameters()]),lr=.04)
    for step in range(steps):
        optimizer.zero_grad()
        diff=last(first(x).relu())-target
        loss=(diff*diff).mean()
        loss.backward()
        optimizer.step()
        if step == 0 or step == steps-1:
            print(f"step={step:03d} MSE={loss.data[0]:.8f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Single-file educational autograd and causal Transformer")
    parser.add_argument("--demo", choices=("train", "regression"), default="train")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads", type=int, default=2)
    args = parser.parse_args()

    if args.demo == "regression":
        regression_demo(args.steps, args.seed)
    else:
        train_demo(args.steps, args.seed, args.heads)


if __name__ == "__main__":
    main()