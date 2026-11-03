"""学习阶段 06：Dispatcher 中独立的 Autograd 处理层。

运行：python experiments/mini_torch.py
补建说明：源自 pytorch6.py；修复 x.meta 属性拼写，保留 redispatch 的教学结构。
"""

from __future__ import annotations

from collections import deque
from math import prod
from typing import Callable, Sequence


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


class CPUBackend:
    """CPU 数值实现：只处理 _Meta，不创建反向节点。"""

    @staticmethod
    def _check_same_shape(x: _Meta, y: _Meta) -> None:
        if x.shape != y.shape:
            raise ValueError(f"需要相同形状，实际为 {x.shape} 和 {y.shape}")

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

    def transpose(self, x: _Meta) -> _Meta:
        """二维转置，返回独立的数据副本，不创建共享存储的视图。"""
        if len(x.shape) != 2:
            raise ValueError("transpose 当前只支持二维矩阵")
        rows, cols = x.shape
        return _Meta(
            [x.data[row * cols + col] for col in range(cols) for row in range(rows)],
            (cols, rows),
        )


class Node:
    """反向边及局部求导接口；节点不保存一次任务的累计梯度。"""

    def __init__(self, edge: list[Node | None], name: str) -> None:
        self.edge = edge
        self.name = name

    def apply(self, grad_out: _Meta) -> tuple[_Meta, ...]:
        raise NotImplementedError("具体节点必须实现自己的反向行为")


class LeafNode(Node):
    """把梯度累加到叶子 Value，类似简化的 AccumulateGrad。"""

    def __init__(self, value: Value) -> None:
        if not value.require_grad:
            raise ValueError("不需要梯度的 Value 不应创建 LeafNode")
        super().__init__(edge=[], name=f"LeafNode({value.name})")
        self.value = value

    def apply(self, grad_out: _Meta) -> tuple[_Meta, ...]:
        if grad_out.shape != self.value.shape:
            raise ValueError("叶子的梯度形状必须与数据相同")
        previous = self.value.grad
        # 如果之前value中没有记录梯度，直接clone，否则对梯度进行累加
        self.value.grad = (
            grad_out.clone() if previous is None else _dispatch.call_backend("add", previous, grad_out)
        )
        return ()


class InnerNode(Node):
    def __init__(
        self,
        edge: list[Node | None],
        backrule: Callable[[_Meta], tuple[_Meta, ...]],
        data: dict[str, _Meta] | None = None,
        name: str = "InnerNode",
    ) -> None:
        super().__init__(edge=edge, name=name)
        self.backrule = backrule
        self.data = {} if data is None else data

    def apply(self, grad_out: _Meta) -> tuple[_Meta, ...]:
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
        self.grad: _Meta | None = None
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

    def __rmul__(self, scalar: float | int) -> Value:
        return self * scalar

    def __matmul__(self, other: Value) -> Value:
        return _dispatch.call("matmul", self, other)

    def backward(self, seed: _Meta | None = None) -> None:
        backward(self, seed)


class Autograd:
    """独立的算子处理层：记录求导信息，再把同一个算子继续派发下去。

    这里负责前向时建立反向图；
    AutogradEngine 负责 backward 时执行图。
    两者不是同一个组件。本版仍没有全局 grad mode 和高阶求导。
    """

    def __init__(self, dispatcher: Dispatch) -> None:
        self.dispatcher = dispatcher

    def apply(
        self, op: str, x: Value, y: Value | float, excluded: frozenset[str]
    ) -> Value:
        # 如果y是value，那么y需不需要梯度取决于y.require_grad
        # 如果y不是value，那么y不需要梯度
        other_requires_grad = y.require_grad if isinstance(y, Value) else False
        require_grad = x.require_grad or other_requires_grad
        node: Node | None = None
        # 具体的回调函数是dispatch中的call_backend  
        raw = self.dispatcher.call_backend

        # 如果不需要梯度，直接redispatch，然后返回新创建的value结点，里面关联了node
        if require_grad:
            if op == "scale":
                if not isinstance(y, float):
                    raise TypeError("scale 的第二个参数必须是 float")
                scalar = y
                node = InnerNode(
                    [x.node],
                    lambda grad_out: (raw("scale", grad_out, scalar),),
                    name=f"ScaleBackward({x.name}, {scalar})",
                )
            else:
                if not isinstance(y, Value):
                    raise TypeError(f"{op} 的第二个参数必须是 Value")
                edges = [x.node, y.node]
                if op == "add":
                    node = InnerNode(
                        edges, lambda grad_out: (grad_out, grad_out),
                        name=f"AddBackward({x.name}, {y.name})",
                    )
                elif op == "sub":
                    node = InnerNode(
                        edges,
                        lambda grad_out: (grad_out, raw("scale", grad_out, -1.0)),
                        name=f"SubBackward({x.name}, {y.name})",
                    )
                elif op in ("mul", "matmul"):
                    saved = {"left": x.meta.clone(), "right": y.meta.clone()}
                    if op == "mul":
                        def mul_backward(grad_out: _Meta) -> tuple[_Meta, ...]:
                            return (
                                raw("mul", grad_out, saved["right"]),
                                raw("mul", grad_out, saved["left"]),
                            )
                        backrule = mul_backward
                    else:
                        def matmul_backward(grad_out: _Meta) -> tuple[_Meta, ...]:
                            return (
                                raw("matmul", grad_out, raw("transpose", saved["right"])),
                                raw("matmul", raw("transpose", saved["left"]), grad_out),
                            )
                        backrule = matmul_backward
                    node = InnerNode(
                        edges, backrule, data=saved,
                        name=f"{op.capitalize()}Backward({x.name}, {y.name})",
                    )
                else:
                    raise NotImplementedError(f"缺少 {op} 的反向规则")

        # 再次派发同一个 op，显式排除本层，避免重新选中 Autograd。
        # 输入仍是 Value，requires_grad 也不变；排除的是处理层，而非输入信息。
        result = self.dispatcher.redispatch(op, x, y, exclude=excluded | {"Autograd"})
        return Value(result.meta, result.name, require_grad, node)


class Dispatch:
    """最小的两层 Dispatcher：Autograd 的优先级高于 CPU。

    call/redispatch 接收 Value，按处理层选择实现。
    call_backend 接收 _Meta，查找 CPU 算子实现，供 CPU 层和原始梯度计算使用。
    本例固定只有 CPU，不从 Tensor 派发键或线程局部状态提取 key。
    """

    def __init__(self, *, trace: bool = False) -> None:
        cpu = CPUBackend()
        self.trace = trace
        # 算子名与具体 CPU 实现分离；新增设备时需要扩展这张表及选择逻辑。
        self.cpu_kernels: dict[str, Callable[..., _Meta]] = {
            "add": cpu.add, "sub": cpu.sub, "mul": cpu.mul,
            "scale": cpu.scale, "matmul": cpu.matmul, "transpose": cpu.transpose,
        }
        self.autograd = Autograd(self)
        self.layers: dict[
            str, Callable[[str, Value, Value | float, frozenset[str]], Value]
        ] = {
            "Autograd": self.autograd.apply,
            "CPU": self._cpu,
        }
        self.priority = ("Autograd", "CPU")

    def call(self, op: str, x: Value, y: Value | float) -> Value:
        return self.redispatch(op, x, y, exclude=frozenset())

    def redispatch(
        self, op: str, x: Value, y: Value | float, *, exclude: frozenset[str]
    ) -> Value:
        if op not in self.cpu_kernels or op == "transpose":
            raise NotImplementedError(f"不支持 Value 算子 {op}")
        for key in self.priority:
            if key not in exclude:
                if self.trace:
                    print(f"Dispatcher: {op} → {key}；排除 {sorted(exclude)}")
                return self.layers[key](op, x, y, exclude)
        raise RuntimeError(f"{op} 没有可用处理层，已排除 {sorted(exclude)}")

    def _cpu(
        self, op: str, x: Value, y: Value | float, excluded: frozenset[str]
    ) -> Value:
        argument = y.meta if isinstance(y, Value) else y
        result = self.call_backend(op, x.meta, argument)
        label = y.name if isinstance(y, Value) else str(y)
        # CPU 层只包装数值，反向节点由外层 Autograd 关联。
        return Value(result, name=f"{op}({x.name}, {label})", require_grad=False)

    # call_backend现在接受的参数只能是底层的meta，
    def call_backend(self, op: str, x: _Meta, y: _Meta | float | None = None) -> _Meta:
        """直接选择 CPU 数值实现，不记录反向图；梯度尚未升级为 Value。"""
        kernel = self.cpu_kernels.get(op)
        if kernel is None:
            raise NotImplementedError(f"CPU 未注册算子 {op}")
        if self.trace:
            print(f"  CPU kernel: {op}")
        if op == "transpose":
            if y is not None:
                raise TypeError("transpose 只接受一个张量")
            return kernel(x)
        if op == "scale":
            if not isinstance(y, (int, float)):
                raise TypeError("scale 需要数值系数")
        elif not isinstance(y, _Meta):
            raise TypeError(f"{op} 需要两个 _Meta 张量")
        return kernel(x, y)


_dispatch = Dispatch()


class AutogradEngine:
    """梯度收齐后统一调用 Node.apply；梯度缓冲区属于本次反向任务。"""

    def run_backward(self, output: Node, seed: _Meta) -> None:
        nodes = self._collect_nodes(output)
        pending = self._compute_dependencies(nodes)
        # 不预填 0.0：每个节点的梯度是不同形状的张量，首份贡献直接复制。
        grad_buffers: dict[Node, _Meta] = {output: seed.clone()}
        ready: deque[Node] = deque([output])

        print("\n=== 反向前：等待的梯度贡献数 ===")
        for node in nodes:
            print(f"{node.name}: {pending[node]} 份")

        while ready:
            node = ready.popleft()
            grad_out = grad_buffers.pop(node)
            self._evaluate_function(node, grad_out, pending, grad_buffers, ready)

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
        grad_out: _Meta,
        pending: dict[Node, int],
        grad_buffers: dict[Node, _Meta],
        ready: deque[Node],
    ) -> None:
        print(f"\n执行 {node.name}：上游梯度 {grad_out}")
        contributions = node.apply(grad_out)
        for slot, (edge, contribution) in enumerate(
            zip(node.edge, contributions, strict=True), start=1
        ):
            if edge is None:
                continue
            previous = grad_buffers.get(edge)
            grad_buffers[edge] = (
                contribution.clone()
                if previous is None
                else _dispatch.call_backend("add", previous, contribution)
            )
            pending[edge] -= 1
            print(
                f"  输入边 {slot} → {edge.name}："
                f"累计 {grad_buffers[edge]}；还等 {pending[edge]} 份"
            )
            if pending[edge] == 0:
                ready.append(edge)


_engine = AutogradEngine()


def backward(output: Value, seed: _Meta | None = None) -> None:
    if not output.require_grad or output.node is None:
        raise RuntimeError("当前 Value 没有可用的反向入口")
    if seed is None:
        if output.meta.numel != 1:
            raise ValueError("多元素输出必须显式提供同形状的上游梯度 seed")
        seed = _Meta.ones_like(output.meta)
    if seed.shape != output.shape:
        raise ValueError(f"seed.shape={seed.shape} 与输出形状 {output.shape} 不一致")
    _engine.run_backward(output.node, seed)


class module:
    def __init__(self, name: str) -> None:
        self.name = name

    def forward(self, input: Value) -> Value:
        return input

    def __call__(self, input: Value) -> Value:
        return self.forward(input)


class myModulde(module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.x = Value(_Meta([1, 2, 3, 4, 5, 6], [2, 3]), name="x")
        self.y = Value(_Meta([2, 3, 4, 5, 6, 7], [3, 2]), name="y")

    def forward(self, input: Value) -> Value:
        # (1,2) @ (2,3) @ (3,2) → (1,2)，随后对整个矩阵数乘。
        hidden = input @ self.x
        hidden.name = "hidden"
        return (hidden @ self.y) * 0.5


def main() -> None:
    print("=== 前向：矩阵乘法与数乘 ===")
    _dispatch.trace = True  # 显示前向与反向的派发路径。
    model = myModulde("m1")
    input = Value(_Meta([1, 2], [1, 2]), name="input", require_grad=False)
    output = model(input)
    print("output:", output.meta)
    # 用全 1 的上游梯度，表示对所有输出元素之和求导。
    output.backward(_Meta.ones_like(output.meta))
    print("\n=== 结果 ===")
    print("input.grad:", input.grad)
    print("x.grad:", model.x.grad)
    print("y.grad:", model.y.grad)


if __name__ == "__main__":
    main()