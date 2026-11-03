"""学习阶段 03：统一 Node.apply 与叶子梯度累加。

运行：python experiments/mini_torch.py
补建说明：源自 pytorch3.py；Engine 临时缓冲区与叶子 .grad 分离。
"""

from __future__ import annotations

from collections import deque
from typing import Callable


class Node:
    """共同接口：反向边 + 接收上游梯度的 apply 方法。"""

    def __init__(self, edge: list[Node | None], name: str) -> None:
        self.edge = edge
        self.name = name

    def apply(self, grad_out: float) -> tuple[float, ...]:
        raise NotImplementedError("具体节点必须实现自己的反向行为")


class LeafNode(Node):
    """相当于简化的 AccumulateGrad：把收到的梯度累加到目标叶子 Value。"""

    def __init__(self, value: Value) -> None:
        if not value.require_grad:
            raise ValueError("不需要梯度的 Value 不应创建 LeafNode")
        super().__init__(edge=[], name=f"LeafNode({value.name})")
        self.value = value

    def apply(self, grad_out: float) -> tuple[float, ...]:
        before = self.value.grad
        self.value.grad += grad_out
        print(
            f"  累加到叶子 {self.value.name}.grad："
            f"{before:g} + {grad_out:g} = {self.value.grad:g}"
        )
        return ()  # 没有更早的反向节点，传播到此结束。


class InnerNode(Node):
    """根据局部求导规则，返回与每条输入边对应的梯度贡献。"""

    def __init__(
        self,
        edge: list[Node | None],
        backrule: Callable[[float], tuple[float, ...]],
        data: dict[str, float] | None = None,
        name: str = "InnerNode",
    ) -> None:
        super().__init__(edge=edge, name=name)
        self.backrule = backrule
        self.data = {} if data is None else data
        # 不保存输出 Value，也不在这里保存一次任务的累计梯度。

    def apply(self, grad_out: float) -> tuple[float, ...]:
        return self.backrule(grad_out)


class Value:
    def __init__(
        self,
        data: float,
        name: str | None = None,
        require_grad: bool = True,
        node: Node | None = None,
    ) -> None:
        if node is not None and not require_grad:
            raise ValueError("关闭梯度的 Value 不应关联反向节点")
        self.data = float(data)
        self.name = name
        self.grad = 0.0
        self.require_grad = require_grad
        self.node: Node | None = node
        if require_grad and node is None:
            self.node = LeafNode(self)

    def __add__(self, other: Value) -> Value:
        require_grad = self.require_grad or other.require_grad
        node: Node | None = None
        if require_grad:
            node = InnerNode(
                edge=[self.node, other.node],
                backrule=lambda grad_out: (grad_out, grad_out),
                name=f"AddBackward({self.name}, {other.name})",
            )
        return Value(
            self.data + other.data,
            name=f"({self.name}+{other.name})",
            require_grad=require_grad,
            node=node,
        )

    def __mul__(self, other: Value) -> Value:
        saved = {"left": self.data, "right": other.data}
        require_grad = self.require_grad or other.require_grad
        node: Node | None = None
        if require_grad:
            node = InnerNode(
                edge=[self.node, other.node],
                data=saved,
                backrule=lambda grad_out: (
                    grad_out * saved["right"],
                    grad_out * saved["left"],
                ),
                name=f"MulBackward({self.name}, {other.name})",
            )
        return Value(
            saved["left"] * saved["right"],
            name=f"({self.name}*{other.name})",
            require_grad=require_grad,
            node=node,
        )

    def backward(self, seed: float = 1.0) -> None:
        backward(self, seed)


class AutogradEngine:
    """统一调度两种 Node；梯度收齐后调用 apply，无需对节点类型做分支。"""

    def run_backward(self, output: Node, seed: float) -> None:
        nodes = self._collect_nodes(output)
        pending = self._compute_dependencies(nodes)
        # 临时梯度属于本次任务，不能沿用上一次反向传播留下的中间梯度。
        grad_buffers = {node: 0.0 for node in nodes}
        grad_buffers[output] = seed
        ready: deque[Node] = deque([output])

        print("\n=== 反向前：等待的梯度贡献数 ===")
        for node in nodes:
            print(f"{node.name}: {pending[node]} 份")

        print("\n=== AutogradEngine：就绪队列调度 ===")
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
        grad_out: float,
        pending: dict[Node, int],
        grad_buffers: dict[Node, float],
        ready: deque[Node],
    ) -> None:
        print(f"\n执行 {node.name}：收到完整上游梯度 {grad_out:g}")
        contributions = node.apply(grad_out)
        for slot, (edge, contribution) in enumerate(
            zip(node.edge, contributions, strict=True), start=1
        ):
            if edge is None:
                continue  # 该输入关闭梯度，但它的数值仍可能用于求另一输入的导数。
            before = grad_buffers[edge]
            grad_buffers[edge] += contribution
            pending[edge] -= 1
            print(
                f"  输入边 {slot} → {edge.name}："
                f"累计 {before:g} + {contribution:g} = {grad_buffers[edge]:g}；"
                f"还等 {pending[edge]} 份"
            )
            if pending[edge] == 0:
                ready.append(edge)


_engine = AutogradEngine()


def backward(output: Value, seed: float = 1.0) -> None:
    if not output.require_grad:
        raise RuntimeError("当前 Value 不需要梯度，不能调用 backward")
    if output.node is None:
        raise RuntimeError("需要梯度的 Value 缺少反向入口")
    _engine.run_backward(output.node, float(seed))


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
        self.x = Value(2, name="x")
        self.y = Value(4, name="y")
        self.z = Value(5, name="z")

    def forward(self, input: Value) -> Value:
        a = input * input
        a.name = "a"
        b = a * self.x
        b.name = "b"
        c = input * self.y * a * b
        c.name = "c"

        return c


def main() -> None:
    x = Value(2, name="x")
    w = Value(3, name="w")
    y = x * x * w + x
    y.backward()
    assert y.data == 14 and x.grad == 13 and w.grad == 4
    print("y=", y.data, "dx=", x.grad, "dw=", w.grad)

if __name__ == "__main__":
    main()