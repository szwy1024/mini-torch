"""学习阶段 00：在 Value 内完成标量自动微分。

运行：python experiments/mini_torch.py
补建说明：按最早的标量设计补建；每次 backward 清空图中梯度，尚未拆出 Engine。
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
        if not self.require_grad:
            raise RuntimeError("当前 Value 不需要梯度")
        nodes, seen, stack = [], set(), [self]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.append(node)
            stack.extend(parent for parent in node.parents if parent.require_grad)
        pending = {node: 0 for node in nodes}
        for node in nodes:
            node.grad = 0.0
            for parent in node.parents:
                if parent.require_grad:
                    pending[parent] += 1
        self.grad = float(seed)
        ready = deque([self])
        while ready:
            node = ready.popleft()
            if node.backward_rule is None:
                continue
            for parent, contribution in zip(node.parents, node.backward_rule(node.grad), strict=True):
                if not parent.require_grad:
                    continue
                parent.grad += contribution
                pending[parent] -= 1
                if pending[parent] == 0:
                    ready.append(parent)





# 引擎本身可以复用；一次反向传播的队列和依赖计数存在 run_backward 的局部变量中。






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