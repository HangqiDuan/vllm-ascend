"""Two-output RoPE row lookup for validated decode positions.

This helper does not cache values. Callers must supply valid, nonnegative model
positions, as produced by input preparation. It preserves destination storage
and leaves inactive rows untouched. Unsupported tensor metadata returns False.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rope_gather_kernel(
    COS, SIN, POS, OUT_COS, OUT_SIN,
    NUM_TOKENS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < WIDTH
    position = tl.load(POS + token).to(tl.int64)
    source = position * WIDTH + column
    cos = tl.load(COS + source, mask=active, other=0)
    sin = tl.load(SIN + source, mask=active, other=0)
    destination = token * WIDTH + column
    tl.store(OUT_COS + destination, cos, mask=active)
    tl.store(OUT_SIN + destination, sin, mask=active)


def try_fused_rope_gather(cos_table, sin_table, positions, out_cos, out_sin):
    tensors = (cos_table, sin_table, positions, out_cos, out_sin)
    if any(t.device.type != 'npu' or not t.is_contiguous() for t in tensors):
        return False
    if any(t.device != cos_table.device for t in tensors):
        return False
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        return False
    if cos_table.ndim != 4 or cos_table.shape[1:3] != (1, 1):
        return False
    if sin_table.shape != cos_table.shape or cos_table.shape[-1] <= 0:
        return False
    if cos_table.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if any(t.dtype != cos_table.dtype for t in (sin_table, out_cos, out_sin)):
        return False
    n = positions.numel()
    if any(t.ndim != 4 or t.shape[1:] != cos_table.shape[1:] or t.shape[0] < n
           for t in (out_cos, out_sin)):
        return False
    if n == 0:
        return True
    width = cos_table.shape[-1]
    _rope_gather_kernel[(n,)](
        cos_table, sin_table, positions, out_cos, out_sin,
        NUM_TOKENS=n, WIDTH=width, BLOCK=triton.next_power_of_2(width),
    )
    return True
