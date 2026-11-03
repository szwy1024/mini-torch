"""学习阶段 01：将调度拆入 AutogradEngine。

运行：python experiments/mini_torch.py
补建说明：源自 pytorch1.py；用共享输入的解析导数验证按边累加。
"""

from __future__ import annotations

from collections import deque
from typing import Callable


class Value:
    def __init__(
        self,
        data: float,
        name: str,
        parents: tuple[Value, ...] = (),
        backward_rule: Callable[[float], tuple[float, ...]] | None = None,
        require_grad : bool  = True
    ):
        self.data = float(data)       # 前向计算的数值
        self.name = name              # 只用于打印，方便对应数学公式
        self.grad = 0.0               # 本次反向传播累计的 d(output)/d(self)
        self.parents = parents        # 有序输入边；x*x 必须保留两条指向 x 的边
        self.backward_rule = backward_rule
        self.require_grad = require_grad

    def __add__(self, other: Value) -> Value:
        def backward_rule(grad_out: float) -> tuple[float, float]:
            return grad_out, grad_out

        return Value(
            self.data + other.data,
            name=f"({self.name}+{other.name})",
            parents=(self, other),
            backward_rule=backward_rule,
            require_grad= self.require_grad or other.require_grad        # 计算返回的结点是否需要梯度
        )

    def __mul__(self, other: Value) -> Value:
        saved_left = self.data
        saved_right = other.data

        def backward_rule(grad_out: float) -> tuple[float, float]:
            return grad_out * saved_right, grad_out * saved_left

        return Value(
            saved_left * saved_right,
            name=f"({self.name}*{other.name})",
            parents=(self, other),
            backward_rule=backward_rule,
            require_grad= self.require_grad or other.require_grad       # 计算返回的结点是否需要梯度   
        )

    def __sub__(self, other:Value)->Value:
        def backward_rule(grad_out:float)-> tuple[float,float]:
            return grad_out , -grad_out

        return Value(
            self.data - other.data,
            name=f"({self.name}-{other.name})",
            parents=(self, other),
            backward_rule=backward_rule,
            require_grad= self.require_grad or other.require_grad        # 计算返回的结点是否需要梯度
        )

    def __truediv__ (self, other:Value)->Value:
        saved_left = self.data
        saved_right = other.data

        def backward_rule(grad_out:float)-> tuple[float,float]: 
            return grad_out / saved_right , - grad_out * saved_left / (saved_right * saved_right)
        
        return Value(
            self.data / other.data,
            name=f"({self.name}/{other.name})",
            parents=(self, other),
            backward_rule=backward_rule,
            require_grad= self.require_grad or other.require_grad        # 计算返回的结点是否需要梯度
        )
        

    def backward(self, seed: float = 1.0) -> None:
        """用户入口：对应 Tensor.backward，只把请求交给模块级接口。"""
        backward(self, seed)


class AutogradEngine:
    """单线程教学引擎：只管理图和调度，不定义加减乘除的导数。

    每次 run_backward 都有独立的 nodes、pending 和 ready。
    为方便观察，仍保留所有中间节点的 grad，并在每次运行前清空。
    这与 PyTorch 默认的“叶子梯度累加、中间梯度不保留”有所不同。
    """

    def run_backward(self, output: Value, seed: float) -> None:
        """执行一次反向传播；对应真实引擎的运行入口。"""
        nodes = self._collect_nodes(output)
        pending = self._compute_dependencies(nodes)

        # 延续原来的教学行为：清空本次图中所有节点的梯度。
        for node in nodes:
            node.grad = 0.0

        print("\n=== 反向前：各节点要等待的梯度贡献数 ===")
        for node in sorted(nodes, key=lambda value: value.name):
            print(f"{node.name}: {pending[node]} 份")

        # 从根节点开始。seed 是传给当前输出的初始上游梯度。
        output.grad = seed
        ready: deque[Value] = deque([output])

        print("\n=== AutogradEngine：就绪队列调度 ===")
        while ready:
            node = ready.popleft()
            self._evaluate_function(node, pending, ready)

    def _collect_nodes(self, output: Value) -> list[Value]:
        """图遍历：只收集参与本次反向传播的节点，共享节点只收集一次。"""
        nodes: list[Value] = []
        seen: set[Value] = set()
        stack: list[Value] = [output]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.append(node)
            stack.extend(parent for parent in node.parents if parent.require_grad)
        return nodes

    def _compute_dependencies(self, nodes: list[Value]) -> dict[Value, int]:
        """依赖统计：每个节点还需要收齐多少条边的梯度贡献。"""
        pending = {node: 0 for node in nodes}
        for node in nodes:
            for parent in node.parents:
                if parent.require_grad:
                    # x*x 的两条输入边都计数，不能把 parents 转成 set。
                    pending[parent] += 1
        return pending

    def _evaluate_function(
        self,
        node: Value,
        pending: dict[Value, int],
        ready: deque[Value],
    ) -> None:
        """执行一个就绪节点的规则，并把梯度贡献传给它的输入节点。"""
        print(f"\n取出 {node.name}：累计梯度 = {node.grad:g}")
        if not node.parents:
            print(f"  {node.name} 是叶子，没有更早的输入节点，传播到此结束。")
            return

        assert node.backward_rule is not None
        # 引擎不必知道这是乘法还是除法，只需调用该节点自己的求导规则。
        contributions = node.backward_rule(node.grad)
        for edge, (parent, contribution) in enumerate(
            zip(node.parents, contributions, strict=True), start=1
        ):
            if not parent.require_grad:
                continue
            before = parent.grad
            parent.grad += contribution
            pending[parent] -= 1
            print(
                f"  {node.name} 的输入边 {edge} → {parent.name}："
                f"贡献 {contribution:g}；累计 {before:g} + {contribution:g}"
                f" = {parent.grad:g}；还等 {pending[parent]} 份"
            )
            if pending[parent] == 0:
                ready.append(parent)
                print(f"    {parent.name} 已收齐梯度，加入就绪队列。")


# 引擎本身可以复用；一次反向传播的队列和依赖计数存在 run_backward 的局部变量中。
_engine = AutogradEngine()


def backward(output: Value, seed: float = 1.0) -> None:
    """模块级接口：对应 torch.autograd.backward，检查并准备参数后调用引擎。"""
    if not output.require_grad:
        raise RuntimeError("当前节点不需要梯度，不能调用 backward 方法")
    _engine.run_backward(output, float(seed))


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

        d = b + c - self.z
        d.name = "d"

        e = d / self.z
        e.name = "e"

        return e

def main() -> None:
    x = Value(2, name="x")
    w = Value(3, name="w")
    y = x * x * w + x
    y.backward()
    assert y.data == 14 and x.grad == 13 and w.grad == 4
    print("y=", y.data, "dx=", x.grad, "dw=", w.grad)



if __name__ == "__main__":
    main()