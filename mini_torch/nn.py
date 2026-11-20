"""Module 注册与基础网络层。"""

from __future__ import annotations

from collections import deque
import math
import random
from typing import Iterator

from .storage import DType, _Meta
from .value import Parameter, Value


class Module:
    _parameters: dict[str, Parameter]
    _modules: dict[str, Module]

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        self.name = name

    def __setattr__(self, name: str, value: object) -> None:
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        if isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameter(self) -> Iterator[Parameter]:
        seen_params: set[int] = set()
        seen_modules: set[int] = set()
        queue: deque[Module] = deque([self])
        while queue:
            module = queue.popleft()
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            for param in module._parameters.values():
                if id(param) not in seen_params:
                    seen_params.add(id(param))
                    yield param
            queue.extend(module._modules.values())

    def forward(self, input: Value, /) -> Value:
        return input

    def __call__(self, input: Value) -> Value:
        return self.forward(input)

    def parameters(self) -> Iterator[Parameter]:
        return self.parameter()

    def __delattr__(self, name: str) -> None:
        object.__delattr__(self, name)
        self._parameters.pop(name, None)
        self._modules.pop(name, None)



class Linear(Module):
    """任意正整数输入/输出维度。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype: DType = "float64",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("Linear")
        self.in_features, self.out_features = in_features, out_features
        if rng is not None:
            generator = rng
        else:
            generator = random
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(
            _Meta(
                [
                    generator.uniform(-bound, bound)
                    for _ in range(in_features * out_features)
                ],
                (in_features, out_features),
                dtype=dtype,
            ),
            name="linear_weight",
        )
        if bias:
            self.bias = Parameter(
                _Meta([0] * out_features, (out_features,), dtype=dtype),
                name="linear_bias",
            )
        else:
            self.bias = None

    def forward(self, x: Value) -> Value:
        result = x @ self.weight
        if self.bias is None:
            return result
        else:
            return result + self.bias.broadcast(result.shape)



class LayerNorm(Module):
    def __init__(
        self, features: int, eps: float = 1e-5, *, dtype: DType = "float64"
    ) -> None:
        super().__init__("LayerNorm")
        self.features, self.eps = features, eps
        self.weight = Parameter(
            _Meta([1] * features, (features,), dtype=dtype), name="gamma"
        )
        self.bias = Parameter(
            _Meta([0] * features, (features,), dtype=dtype), name="beta"
        )

    def forward(self, x: Value) -> Value:
        if x.shape[-1] != self.features:
            raise ValueError("LayerNorm 的输入末维不匹配")
        centered = x - x.mean(-1, keepdim=True).broadcast(x.shape)
        variance = (centered * centered).mean(-1, keepdim=True)
        epsilon = Value(
            _Meta([self.eps] * variance.meta.numel, variance.shape, dtype=variance.dtype),
            require_grad=False,
        )
        normalized = centered / (variance + epsilon).sqrt().broadcast(x.shape)
        return normalized * self.weight.broadcast(x.shape) + self.bias.broadcast(
            x.shape
        )



class Embedding(Module):
    """嵌入层，把输入的tokens序列变成向量"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        dtype: DType = "float64",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("Embedding")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        if rng is not None:
            generator = rng
        else:
            generator = random
        bound = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Parameter(
            _Meta(
                [
                    generator.uniform(-bound, bound)
                    for _ in range(vocab_size * embed_dim)
                ],
                (vocab_size, embed_dim),
                dtype=dtype,
            ),
            name="embedding_weight",
        )

    def forward(self, x: Value) -> Value:
        return x.embedding(self.weight)



class LM_Head(Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        *,
        bias: bool = False,
        dtype: DType = "float64",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("LM_Head")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        if rng is not None:
            generator = rng
        else:
            generator = random
        bound = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Parameter(
            _Meta(
                [
                    generator.uniform(-bound, bound)
                    for _ in range(vocab_size * embed_dim)
                ],
                (embed_dim, vocab_size),
                dtype=dtype,
            ),
            name="lm_head_weight",
        )
        if bias:
            self.bias = Parameter(
                _Meta([0] * vocab_size, (vocab_size,), dtype=dtype), name="lm_head_bias"
            )
        else:
            self.bias = None

    def forward(self, x: Value) -> Value:
        result = x @ self.weight
        if self.bias is None:
            return result
        else:
            return result + self.bias.broadcast(result.shape)

