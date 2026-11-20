"""由 Value 运算组合的损失函数，以及预测和因果掩码辅助函数。"""

from __future__ import annotations

from math import prod
from typing import Sequence

from ._checks import _check_floating_dtype, _check_loss_targets
from .storage import DType, _Meta
from .value import Value


def nll_loss(log_probs: Value, targets: Sequence[int]) -> Value:
    """(N, C) 的对数概率与 N 个类别编号；返回平均负对数似然。"""
    rows, classes = log_probs.shape
    _check_loss_targets(rows, classes, targets)
    offsets = [row * classes + target for row, target in enumerate(targets)]
    indices = Value(_Meta(offsets, (rows,), dtype="int32"), require_grad=False)
    selected = log_probs.reshape((log_probs.meta.numel,)).slice(indices, dim=0)
    return selected.mean() * -1.0



def cross_entropy(logits: Value, targets: Sequence[int]) -> Value:
    """输入 (N, C) 的原始分数，组合 log_softmax 与 nll_loss。"""
    return nll_loss(logits.log_softmax(dim=-1), targets)



def MSE(output: Value, targets: Value) -> Value:
    """输入output和target，求MSE"""
    if output.shape != targets.shape:
        raise ValueError(f"需要相同形状，实际为 {output.shape} 和 {targets.shape}")
    num = prod(output.shape)
    diff = output - targets
    return (diff * diff).mean()



def get_max_index(input: Value) -> Value:
    """返回每行最大值的对应的索引"""
    index: list[int] = []
    rows = input.shape[0]
    cols = input.shape[1]
    for position in range(rows):
        start = position * cols
        row = input.data[start : start + cols]
        index.append(max(range(cols), key=row.__getitem__))

    return Value(_Meta(index, (rows,), dtype="int32"), require_grad=False)



def generate_mask(seq_len: int, *, dtype: DType = "float64") -> Value:
    _check_floating_dtype(dtype)
    result: list[float] = []
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                result.append(float("-inf"))
            else:
                result.append(0.0)
    return Value(_Meta(result, (seq_len, seq_len), dtype=dtype), require_grad=False)

