import torch as th

DTYPE_ALIASES: dict[th.dtype, set[str]] = {
    th.float32: {"float32", "float", "fp32", "f32"},
    th.float16: {"float16", "fp16", "f16"},
    th.bfloat16: {"bfloat16", "bf16"},
    th.float16: {"float16", "fp16", "f16"},
    th.float64: {"float64", "fp64", "f64"},
}
DTYPE_MAP: dict[str, th.dtype] = {
    alias: dtype for dtype, aliases in DTYPE_ALIASES.items() for alias in aliases
}


def get_dtype(dtype_str: str) -> th.dtype:
    dtype = DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Invalid dtype: {dtype_str}")

    return dtype
