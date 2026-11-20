"""教学框架的公共接口；支持 CPU 算子与 Metal float32 矩阵乘法。"""

from .storage import DType, Storage, _Meta
from .value import Parameter, Value
from .autograd import (
    backward, enable_grad, is_grad_enabled, no_grad, set_grad_enabled,
)
from .functional import MSE, cross_entropy, generate_mask, get_max_index, nll_loss
from .nn import Embedding, LayerNorm, Linear, LM_Head, Module
from .models import Attention, FFN, Transformer, TransformerBlock
from .optim import SGD

__all__ = [
    "DType", "Storage", "_Meta", "Value", "Parameter",
    "backward", "no_grad", "enable_grad", "is_grad_enabled", "set_grad_enabled",
    "MSE", "cross_entropy", "nll_loss", "get_max_index", "generate_mask",
    "Module", "Linear", "LayerNorm", "Embedding", "LM_Head",
    "Attention", "FFN", "TransformerBlock", "Transformer", "SGD",
]
