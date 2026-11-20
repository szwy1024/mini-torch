"""数值后端接口。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..storage import _Meta


class Backend:
    def __init__(self, name: str):
        self.name = name

    def get_ability(self) -> dict[str, Callable[..., _Meta]]:
        raise NotImplementedError

    def supports(self, op: str, x: _Meta, y: _Meta | float | None) -> bool:
        """算子已注册后，检查当前输入是否适用于这个后端。"""
        return True

