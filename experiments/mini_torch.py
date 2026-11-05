"""学习阶段 18：max 掩码与稳定 softmax / log_softmax。

运行：python experiments/mini_torch.py
max 并列时选首个位置；softmax 先减去最大值。组合运算保留高阶求导。
纯 Python CPU 教学实现；基础算子经 Dispatcher，组合模块复用 Value 的求导。
"""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from math import exp, log
import math
import random
from typing import Callable, Generator, Sequence
from typing import Iterator


def _normalize_dim(dim: int, ndim: int, *, insert: bool = False) -> int:
    """普通维度范围 [-ndim, ndim-1]；插入位置范围 [-ndim-1, ndim]。"""
    limit = ndim + 1 if insert else max(ndim, 1)
    if not -limit <= dim < limit:
        raise ValueError(f"dim={dim} 超出范围；输入维数={ndim}，insert={insert}")
    return dim + limit if dim < 0 else dim


def _check_same_shape(x: _Meta, y: _Meta) -> None:
    if x.shape != y.shape:
        raise ValueError(f"需要相同形状，实际为 {x.shape} 和 {y.shape}")


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
        if any(dim < 0 for dim in self.shape):
            raise ValueError("shape 不能包含负数")
        self.storage = Storage(data)
        if len(self.data) != self.numel:
            raise ValueError("数据长度与 shape 不匹配")

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


@dataclass(frozen=True)
class OpOptions:
    """在处理层之间传递不可变的算子配置；配置不参与梯度计算。"""

    dim: int | None = None
    size: int | None = None
    keepdim: bool = False
    shape: tuple[int, ...] | None = None


class Backend:
    def __init__(self, name: str):
        self.name = name

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        raise NotImplementedError


class CPUBackend(Backend):
    """CPU 数值实现：只处理 _Meta，不创建反向节点。"""

    def __init__(self, name: str):
        super().__init__(name)

    def add(self, x: _Meta, y: _Meta) -> _Meta:
        _check_same_shape(x, y)
        return _Meta([a + b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def sub(self, x: _Meta, y: _Meta) -> _Meta:
        _check_same_shape(x, y)
        return _Meta([a - b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def mul(self, x: _Meta, y: _Meta) -> _Meta:
        """逐元素乘法，区别于 matmul。"""
        _check_same_shape(x, y)
        return _Meta([a * b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def div(self, x: _Meta, y: _Meta) -> _Meta:
        """同形状逐元素相除；不执行矩阵求逆或广播。"""
        _check_same_shape(x, y)
        return _Meta([a / b for a, b in zip(x.data, y.data, strict=True)], x.shape)

    def log(self, x: _Meta) -> _Meta:
        """逐元素自然对数。"""
        return _Meta([log(value) for value in x.data], x.shape)

    def scale(self, x: _Meta, scalar: float) -> _Meta:
        """矩阵或张量的数乘，scalar 是普通 Python 常数。"""
        return _Meta([value * scalar for value in x.data], x.shape)

    def matmul(self, x: _Meta, y: _Meta) -> _Meta:
        """(m, k) @ (k, n) → (m, n)，仅支持二维矩阵。"""
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise ValueError("matmul 当前只支持两个二维矩阵")
        m, k = x.shape
        other_k, n = y.shape
        if k != other_k:
            raise ValueError(f"矩阵内维不匹配：{x.shape} @ {y.shape}")
        result = [0.0] * (m * n)
        for row in range(m):
            for col in range(n):
                # 行优先：x[row, inner] 和 y[inner, col] 的一维下标。
                result[row * n + col] = sum(
                    x.data[row * k + inner] * y.data[inner * n + col]
                    for inner in range(k)
                )
        return _Meta(result, (m, n))

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
        if dim is not None:
            dim = _normalize_dim(dim, x.ndim)
        if dim is None:
            shape = (1,) * x.ndim if keepdim else ()
            return _Meta([sum(x.data)], shape)

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
                    offset = (outer_index * reduce_size + reduce_index) * inner + inner_index
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
        target = tuple(shape)
        if prod(target) != x.numel:
            raise ValueError(f"reshape 元素数不匹配：{x.shape} → {target}")
        return _Meta(x.data, target)

    def exponential(self, x: _Meta) -> _Meta:
        return _Meta([exp(data) for data in x.data], x.shape)

    def relu(self, x: _Meta) -> _Meta:
        return _Meta([max(data, 0) for data in x.data], x.shape)

    def sqrt(self, x: _Meta) -> _Meta:
        return _Meta([math.sqrt(v) for v in x.data], x.shape)

    def max(self, x: _Meta, dim: int | None = None, *, keepdim: bool = False) -> _Meta:
        return self._max_with_mask(x, dim, keepdim)[0]

    def max_mask(self, x: _Meta, dim: int | None = None, *, keepdim: bool = False) -> _Meta:
        """返回输入形状的首个最大位置掩码；选择结果本身不可微。"""
        return self._max_with_mask(x, dim, keepdim)[1]

    def _max_with_mask(self, x: _Meta, dim: int | None, keepdim: bool) -> tuple[_Meta, _Meta]:
        """返回最大值和首个最大位置掩码；并列最大值选择第一个。"""
        if dim is not None:
            dim = _normalize_dim(dim, x.ndim)
        if dim is None or x.ndim == 0:
            best = max(x.data)
            mask = [0.0] * x.numel
            mask[x.data.index(best)] = 1.0
            shape = (1,) * x.ndim if keepdim else ()
            return _Meta([best], shape), _Meta(mask, x.shape)
        axis = dim
        width = x.shape[axis]
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
        }


class GPUBackend(Backend):
    def __init__(self, name: str):
        super().__init__(name)

    def add(self, x: _Meta, y: _Meta) -> _Meta:
        _check_same_shape(x, y)
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
        super().__init__(edge=[], name=f"LeafNode({value.name})")
        self.value = value

    def apply(self, grad_out: Value) -> tuple[Value, ...]:
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
        return _dispatch.call("matmul", self, other)

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
        """从右对齐各维，长度为 1 的维度可以扩展；组合 reshape 和 expand。"""
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
            reduced = result.shape[:axis] + result.shape[axis + 1:]
            result = result.reshape(reduced).expand(axis, size)
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
        return self.sum(dim, keepdim=keepdim) * (1.0 / count)

    def sqrt(self) -> Value:
        return _dispatch.call("sqrt", self)

    def max(self, dim: int | None = None, *, keepdim: bool = False) -> Value:
        return _dispatch.call("max", self, dim=dim, keepdim=keepdim)

    def softmax(self, dim: int = -1) -> Value:
        axis = _normalize_dim(dim, self.meta.ndim)
        maximum = self.max(axis, keepdim=True)
        shifted = self - maximum.broadcast_to(self.shape)
        numerator = shifted.exp()
        return numerator / numerator.sum(axis, keepdim=True).broadcast_to(self.shape)

    def log_softmax(self, dim: int = -1) -> Value:
        axis = _normalize_dim(dim, self.meta.ndim)
        maximum = self.max(axis, keepdim=True)
        shifted = self - maximum.broadcast_to(self.shape)
        normalizer = shifted.exp().sum(axis, keepdim=True).log()
        return shifted - normalizer.broadcast_to(self.shape)


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
        require_grad = is_grad_enabled() and (x.require_grad or other_requires_grad)
        if op == "max_mask":  # 离散选择不建图。
            require_grad = False
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
            elif op == "max":
                mask = self.dispatcher.call("max_mask", x, dim=options.dim, keepdim=options.keepdim)
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
            elif op == "scale" and isinstance(y, float):
                scalar = y
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (grad_out * scalar,),
                    name=f"ScaleBackward({x.name}, {scalar})",
                )
            elif op == "sum":
                input_shape = x.shape
                axis = options.dim
                # 先将归约轴恢复成长度 1，再广播回输入形状；兼容 keepdim 和标量。
                if axis is None or not input_shape:
                    sum_shape = (1,) * len(input_shape)
                else:
                    sum_shape = input_shape[:axis] + (1,) + input_shape[axis + 1:]

                def sum_backward(grad_out: Value) -> tuple[Value, ...]:
                    return (grad_out.reshape(sum_shape).broadcast_to(input_shape),)

                node = InnerNode([x.node], sum_backward, name=f"SumBackward({x.name})")
            elif op == "expand" and options.dim is not None:
                expand_axis = options.dim
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

            elif isinstance(y, Value):
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
                                grad_out @ saved["right"].T,
                                saved["left"].T @ grad_out,
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
        if op in ("sum", "max", "max_mask"):
            return kernel(x, options.dim, keepdim=options.keepdim)
        if op in ("sqrt", "transpose", "clone", "exponential", "relu", "log"):
            return kernel(x)
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
        target = tuple(shape) if shape is not None else None
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
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.data[:] = (param - param.grad * self.lr).data


def backward(
    output: Value,
    seed: Value | None = None,
    *,
    create_graph: bool = False,
    _trace: bool = False,
) -> None:
    if output.node is None:
        raise RuntimeError("当前 Value 没有可用的反向入口")
    if seed is None:
        if output.meta.numel != 1:
            raise ValueError("多元素输出必须显式提供同形状的上游梯度 seed")
        seed = Value(_Meta.ones_like(output.meta), name="seed", require_grad=False)
    if seed.shape != output.shape:
        raise ValueError(f"seed.shape={seed.shape} 与输出形状 {output.shape} 不一致")
    _engine.run_backward(output.node, seed, create_graph=create_graph, _trace=_trace)


class Parameter(Value):
    """标记：这是模型参数，而不是普通的中间计算结果。"""

    pass


class Module:
    _parameters: dict[str, Parameter]
    _modules: dict[str,Module]

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self,"_modules",{})
        self.name = name

    def __setattr__(self, name: str, value: object) -> None:
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        if isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameter(self) -> Iterator[Parameter]:
        seen_params: set[int] = set()
        seen_modules: set[int] = set()
        queue: deque[Module] = deque([self])
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


    def forward(self, input: Value, /) -> Value:
        return input

    def __call__(self, input: Value) -> Value:
        return self.forward(input)

    def parameters(self) -> Iterator[Parameter]:
        return self.parameter()

    def __delattr__(self, name: str) -> None:
        object.__delattr__(self, name)
        self._parameters.pop(name, None)
        self._modules.pop(name, None)


class Linear(Module):
    """任意正整数输入/输出维度；末维做投影，显式广播偏置。"""
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True,
                 rng: random.Random | None = None) -> None:
        super().__init__("Linear")
        self.in_features, self.out_features = in_features, out_features
        generator = rng if rng is not None else random
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(_Meta(
            [generator.uniform(-bound, bound) for _ in range(in_features * out_features)],
            (in_features, out_features)), name="weight")
        self.bias = Parameter(_Meta([0] * out_features, (out_features,)), name="bias") if bias else None

    def forward(self, x: Value) -> Value:
        result = x @ self.weight
        return result if self.bias is None else result + self.bias.broadcast_to(result.shape)


class LayerNorm(Module):
    def __init__(self, features: int, eps: float = 1e-5) -> None:
        super().__init__("LayerNorm")
        self.features, self.eps = features, eps
        self.weight = Parameter(_Meta([1] * features, (features,)), name="gamma")
        self.bias = Parameter(_Meta([0] * features, (features,)), name="beta")

    def forward(self, x: Value) -> Value:
        if x.shape[-1] != self.features:
            raise ValueError("LayerNorm 的输入末维不匹配")
        centered = x - x.mean(-1, keepdim=True).broadcast_to(x.shape)
        variance = (centered * centered).mean(-1, keepdim=True)
        epsilon = Value(_Meta([self.eps] * variance.meta.numel, variance.shape), require_grad=False)
        normalized = centered / (variance + epsilon).sqrt().broadcast_to(x.shape)
        return normalized * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)


def main() -> None:
    x = Value(_Meta([1000,1001,1002, -1000,-1000,-1000], (2,3)), name="scores")
    p = x.softmax()
    seed = Value(_Meta([1,2,4, 1,2,4], (2,3)), require_grad=False)
    p.backward(seed)
    print("稳定 softmax：", p.data)
    print("scores 梯度：", x.grad)
    tie = Value(_Meta([5,5,1], (3,)))
    tie.max().backward()
    print("并列最大值的梯度：", tie.grad)  # 首个最大位置得到梯度。


if __name__ == "__main__":
    main()
