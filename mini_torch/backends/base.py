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

