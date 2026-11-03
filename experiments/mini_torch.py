"""学习阶段 12：逐元素除法与自然对数。

运行：python experiments/mini_torch.py
补建说明：源自 pytorch12.py；含定义域检查与可继续求导的反向规则。
"""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from math import exp, log
from typing import Callable, Generator, Sequence
from typing import Iterator

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

    def apply(
        self,
        op: str,
        x: Value,
        y: Value | float | None,
        options: OpOptions,
        excluded: frozenset[str],
    ) -> Value:
        argument = y.meta if isinstance(y, Value) else y
        result = self.call_kernel(op, x.meta, argument, options=options)
        if op in ("sum", "expand", "reshape"):
            name = f"{op}({x.name}, {options})"
        else:
            label = y.name if isinstance(y, Value) else str(y)
            name = f"{op}({x.name})" if y is None else f"{op}({x.name}, {label})"
        # CPU 层只包装数值，不决定最终输出是否需要梯度。
        return Value(result, name=name, require_grad=False)

    def call_kernel(
        self, op: str, x: _Meta, y: _Meta | float | None = None, *, options: OpOptions
    ) -> _Meta:
        # 仍固定选 CPU；GPUBackend 只是占位，未实现 Metal 或设备存储。
        kernel = self.kernels["cpu"].get(op)
        if kernel is None:
            raise NotImplementedError(f"CPU 未注册算子 {op}")
        if self.dispatcher._trace:
            print(f"  CPU kernel: {op}")
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
    ) -> Value:
        if op not in self.operator:
            raise NotImplementedError(f"不支持 Value 算子 {op}")
        if not isinstance(x, Value):
            raise TypeError("x 必须是 Value")
        if type(keepdim) is not bool:
            raise TypeError("keepdim 必须是 bool")
        if op != "sum" and keepdim:
            raise TypeError(f"{op} 不接受 keepdim=True")
        if op not in ("sum", "expand") and dim is not None:
            raise TypeError(f"{op} 不接受 dim")
        if op != "expand" and size is not None:
            raise TypeError(f"{op} 不接受 size")
        if op != "reshape" and shape is not None:
            raise TypeError(f"{op} 不接受 shape")

        unary = {
            "transpose",
            "clone",
            "sum",
            "expand",
            "reshape",
            "exponential",
            "relu",
            "log",
        }
        if op in unary:
            if y is not None:
                raise TypeError(f"{op} 只接受一个 Value；dim/size 等配置请使用关键字")
        elif op == "scale":
            if isinstance(y, bool) or not isinstance(y, (int, float)):
                raise TypeError("scale 需要 int 或 float 系数")
            y = float(y)
        elif not isinstance(y, Value):
            raise TypeError(f"{op} 需要两个 Value")

        # 与 no_grad 使用相同的定义域检查，且检查失败时不创建反向图。
        if op == "div":
            if not isinstance(y, Value):
                raise TypeError("div 需要两个 Value")
            _check_div_inputs(x.meta, y.meta)
        elif op == "log":
            _check_log_input(x.meta)

        normalized_dim = dim
        if op == "sum" and dim is not None:
            normalized_dim = _normalize_dim(dim, x.meta.ndim)
        elif op == "expand":
            if dim is None or size is None:
                raise TypeError("expand 必须提供 dim 和 size")
            normalized_dim = _normalize_dim(dim, x.meta.ndim, insert=True)
            _check_size(size)
        target = None
        if op == "reshape":
            if shape is None:
                raise TypeError("reshape 必须提供 shape")
            target = _check_shape(shape)
            if prod(target) != x.meta.numel:
                raise ValueError(f"reshape 元素数不匹配：{x.shape} → {target}")
        options = OpOptions(normalized_dim, size, keepdim, target)
        # 入参验证先于建图；no_grad 与正常模式执行相同的检查。
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


class Optime:
    def __init__(self, params: Iterator[Parameter], lr: float) -> None:
        # 将迭代器转换成列表
        self.params = list(params)

        if lr is None:
            raise RuntimeError("请设置学习率")

        self.lr = lr

    def zero_grad(self):
        """清空梯度"""
        if self.params is None:
            raise RuntimeError("请先初始化参数列表")
        for param in self.params:
            param.grad = None

    def step(self):
        """更新梯度"""
        if self.params is None:
            raise RuntimeError("请先初始化参数列表")
        for param in self.params:
            if param.grad is not None:
                with no_grad():
                    updata = param - param.grad * self.lr
                    param.data[:] = updata.data


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


class module:
    _parameters: dict[str, Parameter]

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_parameters", {})
        self.name = name

    def __setattr__(self, name: str, value: object) -> None:
        parameter = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if parameter is None:
                raise RuntimeError("请先调用super().__init__()")
            parameter[name] = value
        elif parameter is not None:
            parameter.pop(name, None)

        object.__setattr__(self, name, value)

    def parameter(self) -> Iterator[Parameter]:
        seen: set[int] = set()
        for param in self._parameters.values():
            identity = id(param)

            if identity in seen:
                continue

            seen.add(identity)
            yield param

    def forward(self, input: Value) -> Value:
        return input

    def __call__(self, input: Value) -> Value:
        return self.forward(input)


class myModulde(module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.x = Parameter(_Meta([1, 2, -3, -4, 5, -6], [2, 3]), name="x")
        self.y = Parameter(_Meta([2, -3, 4, 5, -6, 7], [3, 2]), name="y")

    def forward(self, input: Value) -> Value:
        # (1,2) @ (2,3) @ (3,2) → (1,2)，随后对整个矩阵数乘。
        self.hidden1 = input @ self.x
        self.hidden2 = self.hidden1.relu()
        self.hidden3 = (self.hidden2 @ self.y) * 0.5
        self.hidden4 = self.hidden3.relu()
        return self.hidden4 @ self.hidden4.T


def main() -> None:
    _dispatch._trace = False

    z = Value(_Meta([1, -2, 3, -4, -5, 6], (2, 3)), name="z")
    loss = (z * z).relu().sum()
    loss.backward(create_graph=True)
    first = z.grad
    assert first is not None
    z.grad = None
    first.sum().backward()
    print("平方和的二阶结果：", z.grad)  # 全部为 2

    # 创建模型
    model = myModulde("m1")
    # 创建输入
    input = Value(_Meta([1, 2], [1, 2]), name="input", require_grad=False)
    # 得到输出

    # 初始化一个优化器
    opt = Optime(model.parameter(), lr=0.001)
    # 清空梯度

    for _ in range(10):

        opt.zero_grad()

        output = model(input)
        print("-------更新前输出------")
        print("output:", output)

        # 反向传播
        output.backward(
            Value(_Meta.ones_like(output.meta), require_grad=False), _trace=False
        )

        # 使用梯度更新权重
        opt.step()
        with no_grad():
            output = model(input)

        print("-------更新后输出------")
        print("output:", output)


if __name__ == "__main__":
    main()