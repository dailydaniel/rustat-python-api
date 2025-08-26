try:
    from .maha import triton_influence
except ImportError:
    triton_influence = None

__all__ = ["triton_influence"]
