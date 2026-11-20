"""数值后端；GPUBackend 当前仅为占位。"""

from .base import Backend
from .cpu import CPUBackend
from .gpu import GPUBackend
from .metal import MetalBackend

__all__ = ["Backend", "CPUBackend", "GPUBackend", "MetalBackend"]
