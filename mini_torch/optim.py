"""参数优化器。"""

from __future__ import annotations

from typing import Iterator

from .autograd import no_grad
from .storage import _Meta
from .value import Parameter, Value


class SGD:
    """只更新不同的叶子参数；保持 Parameter 对象和存储身份不变。"""

    def __init__(self, params: Iterator[Parameter],*, lr: float = 0.01) -> None:
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


class Adam:
    """记录每个参数的梯度均值 m、平方均值 v 和实际更新次数。"""

    def __init__(
        self,
        params: Iterator[Parameter],*,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = list(dict.fromkeys(params))
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m: dict[Parameter, Value] = {}
        self.v: dict[Parameter, Value] = {}
        self.steps: dict[Parameter, int] = {}
        for param in self.params:
            self.m[param] = Value(_Meta.zeros_like(param.meta), require_grad=False)
            self.v[param] = Value(_Meta.zeros_like(param.meta), require_grad=False)
            self.steps[param] = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        beta1, beta2 = self.betas
        with no_grad():
            for param in self.params:
                grad = param.grad
                if grad is None:
                    continue

                self.steps[param] += 1
                step = self.steps[param]

                # 更新历史梯度及梯度平方的滑动平均。
                m = self.m[param] * beta1 + grad * (1.0 - beta1)
                grad_squared = grad * grad
                v = self.v[param] * beta2 + grad_squared * (1.0 - beta2)
                self.m[param] = m
                self.v[param] = v

                # 修正初始 m、v 都为零带来的偏差。
                correction1 = 1.0 - beta1 ** step
                correction2 = 1.0 - beta2 ** step
                m_corrected = m * (1.0 / correction1)
                v_corrected = v * (1.0 / correction2)

                epsilon = Value(
                    _Meta([self.eps] * param.meta.numel, param.shape, dtype=param.dtype),
                    require_grad=False,
                )
                denominator = v_corrected.sqrt() + epsilon
                direction = m_corrected / denominator
                updated = param - direction * self.lr
                memoryview(param.data)[:] = memoryview(updated.data)
