"""学习阶段 04：张量元数据与逐元素数值派发。

运行：python experiments/mini_torch.py
补建说明：第 4 版原稿标量与张量接口混用；依据第 5 版的已完成实现补齐本阶段，仅保留逐元素运算。
"""

from __future__ import annotations

from collections import deque
from math import prod
from typing import Callable, Sequence





class _Meta:
    """数值张量描述：存储 + 形状；不包含自动微分信息。"""

    def __init__(self, data: Sequence[float], shape: Sequence[int]) -> None:
        self.shape = tuple(shape)
        if any(type(dim) is not int or dim < 0 for dim in self.shape):
            raise ValueError("shape 必须由非负整数组成")
        self.data = [float(v) for v in data]
        if len(self.data) != self.numel:
            raise ValueError("数据长度与 shape 不匹配")



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


class Dispatch:
    """统一数值入口；以后可以在这一层选择不同设备的计算实现。"""

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






_dispatch = Dispatch()


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
            grad_out.clone() if previous is None else _dispatch.add(previous, grad_out)
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
    def data(self) -> list[float]:
        return self.meta.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.meta.shape

    # 重载value的运算符，然后在重载运算符的函数中实现反向计算图的构造。
    # 正向计算和反向计算都通过dispatch实现
    def __add__(self, other: Value) -> Value:
        result = _dispatch.add(self.meta, other.meta)
        require_grad = self.require_grad or other.require_grad
        node = None
        if require_grad:
            node = InnerNode(
                [self.node, other.node],
                lambda grad_out: (grad_out, grad_out),
                name=f"AddBackward({self.name}, {other.name})",
            )
        return Value(result, f"({self.name}+{other.name})", require_grad, node)

    def __sub__(self, other: Value) -> Value:
        result = _dispatch.sub(self.meta, other.meta)
        require_grad = self.require_grad or other.require_grad
        node = None
        if require_grad:
            node = InnerNode(
                [self.node, other.node],
                lambda grad_out: (grad_out, _dispatch.scale(grad_out, -1.0)),
                name=f"SubBackward({self.name}, {other.name})",
            )
        return Value(result, f"({self.name}-{other.name})", require_grad, node)

    def __mul__(self, other: Value | float | int) -> Value:
        # 如果是数乘
        if isinstance(other, (int, float)):
            scalar = float(other)
            result = _dispatch.scale(self.meta, scalar)
            node = None
            if self.require_grad:
                node = InnerNode(
                    [self.node],
                    lambda grad_out: (_dispatch.scale(grad_out, scalar),),
                    name=f"ScaleBackward({self.name}, {scalar})",
                )
            return Value(result, f"({self.name}*{scalar})", self.require_grad, node)

        # 矩阵对应元素相乘
        result = _dispatch.mul(self.meta, other.meta)
        require_grad = self.require_grad or other.require_grad
        node = None
        if require_grad:
            # 保存副本：后续修改输入数据，不会改变这次前向对应的求导规则。
            saved = {"left": self.meta.clone(), "right": other.meta.clone()}
            node = InnerNode(
                [self.node, other.node],
                lambda grad_out: (
                    _dispatch.mul(grad_out, saved["right"]),
                    _dispatch.mul(grad_out, saved["left"]),
                ),
                data=saved,
                name=f"MulBackward({self.name}, {other.name})",
            )
        return Value(result, f"({self.name}*{other.name})", require_grad, node)

    def __rmul__(self, scalar: float | int) -> Value:
        return self * scalar



    def backward(self, seed: _Meta | None = None) -> None:
        backward(self, seed)


class AutogradEngine:
    """梯度收齐后统一调用 Node.apply；梯度缓冲区属于本次反向任务。"""

    def run_backward(self, output: Node, seed: _Meta) -> None:
        nodes = self._collect_nodes(output)
        pending = self._compute_dependencies(nodes)
        # 每个节点的梯度是不同形状的张量，首份贡献直接复制。
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
                else _dispatch.add(previous, contribution)
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





def main() -> None:
    x = Value(_Meta([1, 2, 3, 4], (2, 2)), name="x")
    y = x * x + x
    y.backward(_Meta.ones_like(y.meta))
    assert y.data == [2, 6, 12, 20] and x.grad.data == [3, 5, 7, 9]
    print("逐元素结果：", y.data, "梯度：", x.grad.data)


if __name__ == "__main__":
    main()