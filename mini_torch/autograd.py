"""梯度模式、反向节点与 Autograd Engine。"""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable, Generator

from ._checks import _check_same_dtype
from .storage import _Meta

if TYPE_CHECKING:
    from .value import Value

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
        if previous is None:
            self.value.grad = grad_out.clone()
        else:
            self.value.grad = previous + grad_out
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
        if data is None:
            self.data = {}
        else:
            self.data = data

    def apply(self, grad_out: Value) -> tuple[Value, ...]:
        return self.backrule(grad_out)



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
            if previous is None:
                grad_buffers[edge] = contribution
            else:
                grad_buffers[edge] = previous + contribution
            pending[edge] -= 1
            if _trace:
                print(
                    f"  输入边 {slot} → {edge.name}："
                    f"累计 {grad_buffers[edge]}；还等 {pending[edge]} 份"
                )
            if pending[edge] == 0:
                ready.append(edge)


_engine = AutogradEngine()


def backward(
    output: Value,
    seed: Value | None = None,
    *,
    create_graph: bool = False,
    _trace: bool = False,
) -> None:
    # Value 在运行时才导入，节点的类型注解不需要加载它。
    from .value import Value

    if output.node is None:
        raise RuntimeError("当前 Value 没有可用的反向入口")
    if seed is None:
        if output.meta.numel != 1:
            raise ValueError("多元素输出必须显式提供同形状的上游梯度 seed")
        seed = Value(_Meta.ones_like(output.meta), name="seed", require_grad=False)
    if seed.shape != output.shape:
        raise ValueError(f"seed.shape={seed.shape} 与输出形状 {output.shape} 不一致")
    _check_same_dtype(seed.meta, output.meta)
    _engine.run_backward(output.node, seed, create_graph=create_graph, _trace=_trace)

