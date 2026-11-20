"""注意力、前馈网络与单头 Transformer 模型。"""

from __future__ import annotations

import math
import random

from .functional import generate_mask
from .nn import Embedding, LayerNorm, Linear, LM_Head, Module
from .storage import DType, _Meta
from .value import Value


class Attention(Module):
    """单头自注意力，输入和输出形状均为 (seq_len, embed_dim)。"""

    def __init__(
        self,
        embed_dim: int,
        *,
        bias: bool = False,
        dtype: DType = "float64",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("Attention")
        self.embed_dim = embed_dim
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, rng=rng)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, rng=rng)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, rng=rng)
        

    def forward(self, x: Value) -> Value:
        if x.shape[-1] != self.embed_dim:
            raise ValueError("Attention 的输入末维不匹配")
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        scale = 1.0 / math.sqrt(self.embed_dim)
        scores = (query @ key.T) * scale
        causal_mask = generate_mask(scores.shape[0], dtype=scores.dtype)
        scores = scores + causal_mask
        attention_weights = scores.softmax(dim=-1)
        return attention_weights @ value



class FFN(Module):
    """逐 token 的前馈网络：input_dim → hidden_dim → output_dim。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        bias1: bool = False,
        bias2: bool = False,
        dtype: DType = "float64",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("FFN")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear1 = Linear(input_dim, hidden_dim, bias=bias1, dtype=dtype, rng=rng)
        self.linear2 = Linear(hidden_dim, output_dim, bias=bias2, dtype=dtype, rng=rng)

    def forward(self, x: Value) -> Value:
        if x.shape[-1] != self.input_dim:
            raise ValueError("FFN 的输入末维不匹配")
        return self.linear2(self.linear1(x).relu())



class TransformerBlock(Module):
    """单头 Pre-Norm Transformer 层，输入和输出均为 (seq_len, embed_dim)。"""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        *,
        eps: float = 1e-5,
        attention_bias: bool = False,
        fnn_bias1: bool = False,
        fnn_bias2: bool = False,
        dtype: DType = "float64",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("TransformerBlock")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.norm1 = LayerNorm(embed_dim, eps=eps, dtype=dtype)
        self.norm2 = LayerNorm(embed_dim, eps=eps, dtype=dtype)
        self.attention = Attention(embed_dim, bias=attention_bias, dtype=dtype, rng=rng)
        self.ffn = FFN(
            embed_dim,
            hidden_dim,
            embed_dim,
            bias1=fnn_bias1,
            bias2=fnn_bias2,
            dtype=dtype,
            rng=rng,
        )

    def forward(self, x: Value) -> Value:
        if x.shape[-1] != self.embed_dim:
            raise ValueError("TransformerBlock 的输入末维不匹配")
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



class Transformer(Module):
    """transformer，包含一个嵌入层，三个transformerBlock，一个输出头"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        *,
        eps: float = 1e-5,
        attention_bias: bool = False,
        fnn_bias1: bool = False,
        fnn_bias2: bool = False,
        lm_bias: bool = False,
        max_sequence_len: int = 100,
        dtype: DType = "float64",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("Transformer")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.tokens_embedding = Embedding(vocab_size, embed_dim, dtype=dtype, rng=rng)
        self.max_sequence_len = max_sequence_len
        self.position_embedding = Embedding(
            max_sequence_len, embed_dim, dtype=dtype, rng=rng
        )
        self.transformer_block1 = TransformerBlock(
            embed_dim,
            hidden_dim,
            eps=eps,
            attention_bias=attention_bias,
            fnn_bias1=fnn_bias1,
            fnn_bias2=fnn_bias2,
            dtype=dtype,
            rng=rng,
        )
        self.transformer_block2 = TransformerBlock(
            embed_dim,
            hidden_dim,
            eps=eps,
            attention_bias=attention_bias,
            fnn_bias1=fnn_bias1,
            fnn_bias2=fnn_bias2,
            dtype=dtype,
            rng=rng,
        )
        self.transformer_block3 = TransformerBlock(
            embed_dim,
            hidden_dim,
            eps=eps,
            attention_bias=attention_bias,
            fnn_bias1=fnn_bias1,
            fnn_bias2=fnn_bias2,
            dtype=dtype,
            rng=rng,
        )
        self.layernorm = LayerNorm(embed_dim, eps=eps, dtype=dtype)
        self.lm_head = LM_Head(embed_dim, vocab_size, bias=lm_bias, dtype=dtype, rng=rng)

    def forward(self, x: Value) -> Value:
        seq_len = x.shape[0]
        position_ids = Value(
            _Meta(list(range(seq_len)), (seq_len,), dtype="int32"),
            require_grad=False,
        )
        embedding = self.tokens_embedding(x) + self.position_embedding(position_ids)
        hidden1 = self.transformer_block1(embedding)
        hidden2 = self.transformer_block2(hidden1)
        hidden3 = self.transformer_block3(hidden2)
        final_norm = self.layernorm(hidden3)
        output = self.lm_head(final_norm)
        return output

