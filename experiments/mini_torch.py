"""学习阶段 09：Parameter 注册与 SGD 优化器。

运行：python experiments/mini_torch.py
补建说明：源自 pytorch9.py；同一参数的多个属性按对象身份去重，避免重复更新。
"""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from math import prod
from typing import Callable,Generator, Sequence
from typing import Iterator

# ContextVar 保存当前执行上下文的状态，嵌套或异常退出时用 token 恢复。
_grad_enabled: ContextVar[bool] = ContextVar("grad_enabled", default=True)


def is_grad_enabled() -> bool:
    return _grad_enabled.get()


@contextmanager
def set_grad_enabled(enabled: bool) -> Generator[None,None,None]:
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

class Backend:
        def __init__(self,name:str):
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

    def get_ability(self)->dict[str,Callable[...,_Meta]]:
        return {
                                    "add": self.add, "sub": self.sub, "mul": self.mul,
                                    "scale": self.scale, "matmul": self.matmul,
                                    "transpose": self.transpose, "clone": self.clone,
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

    def get_ability(self)->dict[str,Callable[...,_Meta]]:
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

    def __rmul__(self, scalar: float | int) -> Value:
        return self * scalar

    def __matmul__(self, other: Value) -> Value:
        return _dispatch.call("matmul", self, other)

    def clone(self) -> Value:
        return _dispatch.call("clone", self)

    def transpose(self) -> Value:
        return _dispatch.call("transpose", self)

    # 可以直接使用.T进行矩阵转置运算
    @property
    def T(self) -> Value:
        return self.transpose()

    # create_graph 表示是否在反向计算过程中再次构造反向计算图
    def backward(
        self, seed: Value | None = None, *, create_graph: bool = False,_trace:bool = False
    ) -> None:
        backward(self, seed, create_graph=create_graph,_trace = _trace)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, shape={self.shape}, require_grad={self.require_grad})"


class Autograd:
    """前向和反向的 Value 运算共用此层，是否建图由当前梯度模式决定。"""

    def __init__(self, dispatcher: Dispatch) -> None:
        self.dispatcher = dispatcher

    # excluded表示需要排除的处理流程
    def apply(
        self, op: str, x: Value, y: Value | float | None, excluded: frozenset[str]
    ) -> Value:
        # 只有value类型可能需要梯度，纯数值不关联梯度信息
        other_requires_grad = y.require_grad if isinstance(y, Value) else False
        # 查询当前上下文是否开启了梯度。
        require_grad = is_grad_enabled() and (x.require_grad or other_requires_grad)
        node: Node | None = None
        if self.dispatcher._trace:
            print(f"  Autograd: grad_enabled={is_grad_enabled()}, record={require_grad}")

        if require_grad:
            if op == "clone":
                node = InnerNode(
                    [x.node], lambda grad_out: (grad_out,),
                    name=f"CloneBackward({x.name})",
                )
            elif op == "transpose":
                node = InnerNode(
                    [x.node], lambda grad_out: (grad_out.T,),
                    name=f"TransposeBackward({x.name})",
                )
            elif op == "scale":
                if not isinstance(y, float):
                    raise TypeError("scale 的第二个参数必须是 float")
                scalar = y
                node = InnerNode(
                    [x.node], lambda grad_out: (grad_out * scalar,),
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
                        edges, lambda grad_out: (grad_out, grad_out * -1.0),
                        name=f"SubBackward({x.name}, {y.name})",
                    )
                elif op in ("mul", "matmul"):
                    # 保留前向快照，同时保留它到原输入的求导关系。
                    # x.meta.clone() 只有数值；x.clone() 则通过 CloneBackward 连接 x。
                    # 这是教学策略；真实 PyTorch 通常保存张量并做版本检查。
                    saved = {"left": x.clone(), "right": y.clone()}
                    if op == "mul":
                        def mul_backward(grad_out: Value) -> tuple[Value, ...]:
                            return (grad_out * saved["right"], grad_out * saved["left"])
                        backrule = mul_backward
                    else:
                        def matmul_backward(grad_out: Value) -> tuple[Value, ...]:
                            return (grad_out @ saved["right"].T, saved["left"].T @ grad_out)
                        backrule = matmul_backward
                    node = InnerNode(
                        edges, backrule, data=saved,
                        name=f"{op.capitalize()}Backward({x.name}, {y.name})",
                    )
                else:
                    raise NotImplementedError(f"缺少 {op} 的反向规则")

        # 排除 Autograd 只影响本次算子的后续派发，不改变当前梯度模式。
        result = self.dispatcher.redispatch(op, x, y, exclude=excluded | {"Autograd"})
        return Value(result.meta, result.name, require_grad, node)


class ExecuteLayer:
    def __init__(self,dispatcher:Dispatch):
        self.dispatcher = dispatcher
        cpu = CPUBackend("cpu")
        gpu = GPUBackend("gpu")
        self.kernels: dict[str, dict[str,Callable[..., _Meta]]] = {"cpu":cpu.get_ability() ,
                                                                   "gpu":gpu.get_ability()}
                                
    def apply(
            self, op: str, x: Value, y: Value | float | None, excluded: frozenset[str]
        ) -> Value:
            argument = y.meta if isinstance(y, Value) else y
            result = self.call_kernel(op, x.meta, argument)
            label = y.name if isinstance(y, Value) else str(y)
            name = f"{op}({x.name})" if y is None else f"{op}({x.name}, {label})"
            # 中间计算节点默认不保存梯度
            return Value(result, name=name, require_grad=False)
    
    def call_kernel(self, op: str, x: _Meta, y: _Meta | float | None = None) -> _Meta:
        """仅负责调用backend数值实现；不是反向规则的直接入口。"""
        # 后期根据数据的所在位置选择具体的后端，现在默认cpu
        kernel = self.kernels["cpu"].get(op)
        if kernel is None:
            raise NotImplementedError(f"CPU 未注册算子 {op}")
        if self.dispatcher._trace:
            print(f"  CPU kernel: {op}")
        if op in ("transpose", "clone"):
            if y is not None:
                raise TypeError(f"{op} 只接受一个张量")
            return kernel(x)
        if op == "scale":
            if not isinstance(y, (int, float)):
                raise TypeError("scale 需要数值系数")
        elif not isinstance(y, _Meta):
            raise TypeError(f"{op} 需要两个 _Meta 张量")
        return kernel(x, y)

    def get_kernels(self)->dict[str,dict[str,Callable[...,_Meta]]]:
        return self.kernels


class Dispatch:
    """前向和反向的 Value 运算都经过同一套 Autograd → CPU 派发。"""

    def __init__(self, *, _trace: bool = False) -> None:
        self._trace = _trace
        #目前支持的算子操作
        self.operator = ( "add", "sub", "mul", "scale", "matmul","transpose", "clone")
        self.autograd = Autograd(self)
        self.executeLayer = ExecuteLayer(self)
        self.kernels: dict[str, dict[str,Callable[..., _Meta]]] = self.executeLayer.get_kernels()
        self.layers: dict[
            str, Callable[[str, Value, Value | float | None, frozenset[str]], Value]
        ] = {
            "Autograd": self.autograd.apply,
            "CPU": self.executeLayer.apply,
        }
        self.priority = ("Autograd", "CPU")

    def call(self, op: str, x: Value, y: Value | float | None = None) -> Value:
        return self.redispatch(op, x, y, exclude=frozenset())

    def redispatch(
        self, op: str, x: Value, y: Value | float | None = None, *, exclude: frozenset[str]
    ) -> Value:
        if op not in self.operator:
            raise NotImplementedError(f"不支持 Value 算子 {op}")
        for key in self.priority:
            if key not in exclude:
                if self._trace:
                    print(f"Dispatcher: {op} → {key}；排除 {sorted(exclude)}")
                return self.layers[key](op, x, y, exclude)
        raise RuntimeError(f"{op} 没有可用处理层，已排除 {sorted(exclude)}")

    
_dispatch = Dispatch()


class AutogradEngine:
    """梯度收齐后统一调用 Node.apply；梯度缓冲区属于本次反向任务。"""

    def run_backward(self, output: Node, seed: Value, *, create_graph: bool = False,_trace:bool = False) -> None:
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
                self._evaluate_function(node, grad_out, pending, grad_buffers, ready,_trace)

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
        _trace :bool
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
    def __init__(self,params: Iterator[Parameter],lr: float ) -> None:
        # 将迭代器转换成链表
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
    output: Value, seed: Value | None = None, *, create_graph: bool = False,_trace:bool = False,
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
    _engine.run_backward(output.node, seed, create_graph=create_graph,_trace=_trace)

class Parameter(Value):
    """标记：这是模型参数，而不是普通的中间计算结果。"""
    pass

class module:
    _parameters: dict[str,Parameter]

    def __init__(self, name: str) -> None:
        object.__setattr__(self,"_parameters",{})
        self.name = name

    def __setattr__(self, name: str, value: object) -> None:
        parameter = self.__dict__.get("_parameters")
        if isinstance(value,Parameter):
            if parameter is None:
                raise RuntimeError("请先调用super().__init__()")
            parameter[name] = value
        elif parameter is not None:
            parameter.pop(name,None)

        object.__setattr__(self,name,value)

    def parameter(self) -> Iterator[Parameter]:
        return iter(dict.fromkeys(self._parameters.values()))

    def forward(self, input: Value) -> Value:
        return input
    
    def __call__(self, input: Value) -> Value:
        return self.forward(input)
    
    
class myModulde(module):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.x = Parameter(_Meta([1, 2, 3, 4, 5, 6], [2, 3]), name="x")
        self.y = Parameter(_Meta([2, 3, 4, 5, 6, 7], [3, 2]), name="y")

    def forward(self, input: Value) -> Value:
        # (1,2) @ (2,3) @ (3,2) → (1,2)，随后对整个矩阵数乘。
        self.hidden1 = input @ self.x
        self.hidden2 =  (self.hidden1 @ self.y) * 0.5
        return self.hidden2 @ self.hidden2.T


def main() -> None:
    _dispatch._trace = False

    print("=== 原来的矩阵示例：普通 backward 不记录梯度计算 ===")
    # 创建模型
    model = myModulde("m1")
    # 创建输入
    input = Value(_Meta([1, 2], [1, 2]), name="input", require_grad=False)
    # 得到输出
    
    # 初始化一个优化器
    opt = Optime(model.parameter(),lr = 0.001)
    # 清空梯度

    for _ in range(10):

        opt.zero_grad()

        output = model(input)
        print("-------更新前输出------")
        print("output:", output)

        # 反向传播
        output.backward(Value(_Meta.ones_like(output.meta), require_grad=False),_trace = False)

        # 使用梯度更新权重
        opt.step()
        with no_grad():
            output = model(input)

        print("-------更新后输出------")
        print("output:", output)


if __name__ == "__main__":
    main()