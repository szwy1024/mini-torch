"""参数优化器。"""

from __future__ import annotations

from typing import Iterator

from .autograd import no_grad
from .value import Parameter


class SGD:
    """只更新不同的叶子参数；保持 Parameter 对象和存储身份不变。"""

    def __init__(self, params: Iterator[Parameter], lr: float = 0.01) -> None:
        self.params = list(dict.fromkeys(params))
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with no_grad():
            for param in self.params:
                if param.grad is not None:
                    updated = param - param.grad * self.lr
                    memoryview(param.data)[:] = memoryview(updated.data)

