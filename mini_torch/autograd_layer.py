"""算子的 Autograd 包装层：保存前向信息并建立反向节点。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .autograd import InnerNode, Node, is_grad_enabled, no_grad
from .storage import _Meta
from .value import Value

if TYPE_CHECKING:
    from .dispatcher import Dispatch, OpOptions


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
        if isinstance(y, Value):
            other_requires_grad = y.require_grad
        else:
            other_requires_grad = False
        require_grad = is_grad_enabled() and (x.require_grad or other_requires_grad)
        if op == "max_mask":  # 离散选择不建图。
            require_grad = False
        if op == "embedding":
            require_grad = is_grad_enabled() and other_requires_grad
        if op in ("slice", "scatter"):
            require_grad = is_grad_enabled() and x.require_grad
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
                mask = self.dispatcher.redispatch(
                    "max_mask", x, options=options, exclude=excluded | {"Autograd"}
                )
                if options.dim is None or x.meta.ndim == 0:
                    expanded_shape = (1,) * x.meta.ndim
                else:
                    axis = options.dim
                    expanded_shape = x.shape[:axis] + (1,) + x.shape[axis + 1 :]

                def max_backward(g: Value) -> tuple[Value, ...]:
                    return (g.reshape(expanded_shape).broadcast(x.shape) * mask,)

                node = InnerNode(
                    [x.node], max_backward, data={"mask": mask}, name="MaxBackward"
                )
            elif op == "sqrt":
                saved = {"input": x.clone()}
                node = InnerNode(
                    [x.node],
                    lambda g: (g / (saved["input"].sqrt() * 2.0),),
                    data=saved,
                    name=f"SqrtBackward({x.name})",
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
                mask_data = []
                for value in x.data:
                    if value > 0:
                        mask_data.append(1.0)
                    else:
                        mask_data.append(0.0)
                mask = Value(
                    _Meta(mask_data, x.shape, dtype=x.dtype),
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
                    sum_shape = input_shape[:axis] + (1,) + input_shape[axis + 1 :]

                def sum_backward(grad_out: Value) -> tuple[Value, ...]:
                    return (grad_out.reshape(sum_shape).broadcast(input_shape),)

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
            elif op == "embedding":
                if isinstance(y, Value):
                    weight_shape = y.shape
                    with no_grad():
                        saved = {"indices": x.reshape((x.meta.numel,)).clone()}

                    def embedding_backward(grad_out: Value) -> tuple[Value, ...]:
                        indices = saved["indices"]
                        rows = grad_out.reshape((indices.meta.numel, weight_shape[-1]))
                        return (rows.scatter(indices, size=weight_shape[-2], dim=0),)

                    node = InnerNode(
                        [y.node],
                        embedding_backward,
                        data=saved,
                        name=f"EmbeddingBackward({x.name},{y.name})",
                    )
            elif op == "slice" and isinstance(y, Value) and options.dim is not None:
                axis = options.dim
                size = x.shape[axis]
                with no_grad():
                    saved = {"indices": y.clone()}

                def slice_backward(grad_out: Value) -> tuple[Value, ...]:
                    return (grad_out.scatter(saved["indices"], size=size, dim=axis),)

                node = InnerNode(
                    [x.node],
                    slice_backward,
                    data=saved,
                    name=f"SliceBackward({x.name}, {y.name})",
                )
            elif op == "scatter" and isinstance(y, Value) and options.dim is not None:
                axis = options.dim
                with no_grad():
                    saved = {"indices": y.clone()}

                def scatter_backward(grad_out: Value) -> tuple[Value, ...]:
                    return (grad_out.slice(saved["indices"], dim=axis),)

                node = InnerNode(
                    [x.node],
                    scatter_backward,
                    data=saved,
                    name=f"ScatterBackward({x.name}, {y.name})",
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
                    # x.meta.clone() 只有数值
                    # x.clone() 通过 CloneBackward 连接 x。
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

