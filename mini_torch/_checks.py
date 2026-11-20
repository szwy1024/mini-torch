"""各层复用的形状、类型与索引检查。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .storage import DType, _Meta


def _check_same_dtype(x: _Meta, y: _Meta) -> None:
    if x.dtype != y.dtype:
        raise TypeError(f"需要相同 dtype，实际为 {x.dtype} 和 {y.dtype}")



def _check_floating_dtype(dtype: DType) -> None:
    if dtype == "int32":
        raise TypeError("此操作需要 float32 或 float64")



def _check_indices(indices: _Meta) -> None:
    if indices.dtype != "int32":
        raise TypeError("索引需要 int32")



def _normalize_dim(dim: int, ndim: int, *, insert: bool = False) -> int:
    """普通维度范围 [-ndim, ndim-1]；插入位置范围 [-ndim-1, ndim]。"""
    if insert:
        limit = ndim + 1
    else:
        limit = max(ndim, 1)
    if not -limit <= dim < limit:
        raise ValueError(f"dim={dim} 超出范围；输入维数={ndim}，insert={insert}")
    if dim < 0:
        return dim + limit
    else:
        return dim



def _check_same_shape(x: _Meta, y: _Meta) -> None:
    if x.shape != y.shape:
        raise ValueError(f"需要相同形状，实际为 {x.shape} 和 {y.shape}")



def _check_loss_targets(rows: int, classes: int, targets: Sequence[int]) -> None:
    if len(targets) != rows:
        raise ValueError("每一行输出需要对应一个标签")
    if any(target < 0 or target >= classes for target in targets):
        raise IndexError("标签超出类别范围")

