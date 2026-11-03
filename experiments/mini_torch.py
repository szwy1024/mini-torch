"""学习阶段 02：分离 Value 与反向 Node。

运行：python experiments/mini_torch.py
补建说明：源自 pytorch2.py；修复 right 拼写、require_grad 与输出 node 关联，清空每次内部节点梯度。
"""

from __future__ import annotations

from collections import deque
from typing import Callable


class Node:
    def __init__(self,
        value : Value|None,
    ) -> None: 
        self.value = value

# 叶子结点，负责将梯度传递给value
class LeafNode(Node):
    def __init__(self,
        value : Value|None,
    ) -> None: 
        super().__init__(value)

# 中间结点，负责保存收集梯度
class InnerNode(Node):
    def __init__(self,
        edge : list[Node|None],                          # 反向边
        value : Value|None,
        backrule : Callable[[float],tuple[float,...]],   # 反向求导的方法
        data : dict[str,float] | None = None             # 需要的数据
    ) -> None: 
        super().__init__(value)
        self.edge = edge
        self.backrule = backrule
        self.data = data   
        self.grad = 0.0

class Value:
    def __init__(self, data: float, name: str | None = None,
                 require_grad: bool = True, node: Node | None = None):
        self.data = float(data)
        self.name = name
        self.grad = 0.0
        self.require_grad = require_grad
        self.node = node if node is not None else (LeafNode(self) if require_grad else None)

    def __add__(self, other: Value) -> Value:
        left, right = self.data, other.data
        requires = self.require_grad or other.require_grad
        value = Value(self.data + other.data, name="result", require_grad=requires)
        if requires:
            value.node = InnerNode([self.node, other.node], value,
                                   lambda g: (g, g))
        return value

    def __mul__(self, other: Value) -> Value:
        left, right = self.data, other.data
        requires = self.require_grad or other.require_grad
        value = Value(left * right, name="result", require_grad=requires)
        if requires:
            value.node = InnerNode([self.node, other.node], value,
                                   lambda g: (g * right, g * left))
        return value

    
    def backward(self,seed: float = 1.0) -> None:
        """用户入口：对应 Tensor.backward，只把请求交给模块级接口。"""
        backward(self,seed)


class AutogradEngine:
    """单线程教学引擎：只管理图和调度，不定义加减乘除的导数。

    每次 run_backward 都有独立的 nodes、pending 和 ready。
    为方便观察，仍保留所有中间节点的 grad，并在每次运行前清空。
    这与 PyTorch 默认的“叶子梯度累加、中间梯度不保留”有所不同。
    """

    def run_backward(self, output: Node, seed: float) -> None:
        """执行一次反向传播；对应真实引擎的运行入口。"""
        if isinstance(output, LeafNode):
            raise RuntimeError("无法对叶子结点调用反向传播")

        if isinstance(output,InnerNode) :
            nodes = self._collect_nodes(output)
            pending = self._compute_dependencies(nodes)
            for n in nodes:
                if isinstance(n, InnerNode):
                    n.grad = 0.0

            # 从根节点开始。seed 是传给当前输出的初始上游梯度。
            output.grad = seed
            ready: deque[Node] = deque([output])

            print("\n=== AutogradEngine：就绪队列调度 ===")
            while ready:
                node = ready.popleft()
                self._evaluate_function(node, pending, ready)

    def _collect_nodes(self, output: InnerNode) -> list[Node]:
        """图遍历：只收集参与本次反向传播的节点，共享节点只收集一次。"""
        nodes: list[Node] = []
        seen: set[Node] = set()
        stack: list[Node] = [output]
        while stack:
            node = stack.pop()
            if node in seen :
                continue
            seen.add(node)
            nodes.append(node)
            # 获取node结点关联的边
            if isinstance(node,InnerNode):
                stack.extend([edge for edge in node.edge if edge is not None])
        return nodes

    def _compute_dependencies(self, nodes: list[Node]) -> dict[Node, int]:
        """依赖统计：每个节点还需要收齐多少条边的梯度贡献。"""
        pending = {node: 0 for node in nodes}
        for node in nodes:
            if isinstance(node,InnerNode):
                for edge in node.edge:
                    if edge is not None:
                        pending[edge] += 1
        return pending

    def _evaluate_function(
        self,
        node: Node,
        pending: dict[Node, int],
        ready: deque[Node]
    ) -> None:
        """执行一个就绪节点的规则，并把梯度贡献传给它的输入节点。"""
        
        # 引擎不必知道这是乘法还是除法，只需调用该节点自己的求导规则。
        if isinstance(node,InnerNode):
            contributions = node.backrule(node.grad)
            for edge, (edge, contribution) in enumerate(
                zip(node.edge, contributions, strict=True), start=1
            ):
                if isinstance(edge,InnerNode):
                    edge.grad += contribution
                    pending[edge] -= 1
                
                    if pending[edge] == 0:
                        ready.append(edge)
                if isinstance(edge ,LeafNode):
                    edge.value.grad +=contribution 
        
# 引擎本身可以复用；一次反向传播的队列和依赖计数存在 run_backward 的局部变量中。
_engine = AutogradEngine()


def backward(output: Value, seed: float = 1.0) -> None:
    """模块级接口：对应 torch.autograd.backward，检查并准备参数后调用引擎。"""
    if not output.require_grad:
        raise RuntimeError("当前节点不需要梯度，不能调用 backward 方法")
    _engine.run_backward(output.node, float(seed))


class module:
    def __init__(
        self,
        name:str,
    ):
        self.name=name

    # 继承的模块中需要重写的前向计算方法
    def forward(self,input:Value)->Value:
        return input

    # 可以直接使用模块名调用
    def __call__(self, input:Value) -> Value:
        return self.forward(input)

    
# 定义自己的神经网络模型
class myModulde(module):
    def __init__(self, name: str):
        super().__init__(name)
        self.x = Value(10,name="x")
        self.y = Value(4,name="y")
        self.z = Value(5,name="z")

    def forward(self, input: Value):
        a = input * input
        a.name = "a"

        b = a * self.x
        b.name = "b"

        c = input * self.y
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