"""Experimental two-view gather; no product integration yet."""

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.block_table import _load_ptr


@triton.jit
def _fused_gather_kernel(
    global_idx,
    local_idx,
    src_ptrs,
    global_out_ptrs,
    local_out_ptrs,
    strides,
    num_blocks_ptr,
    num_blocks_stride,
    GLOBAL_REQS: tl.constexpr,
    LOCAL_REQS: tl.constexpr,
    GLOBAL_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    combined_row = tl.program_id(1)
    if combined_row < GLOBAL_PAD:
        row = combined_row
        mapping = global_idx
        output_ptrs = global_out_ptrs
        num_reqs = GLOBAL_REQS
    else:
        row = combined_row - GLOBAL_PAD
        mapping = local_idx
        output_ptrs = local_out_ptrs
        num_reqs = LOCAL_REQS

    stride = tl.load(strides + group)
    dst = _load_ptr(output_ptrs + group, tl.int32) + row * stride
    if row >= num_reqs:
        # Padded rows clear the full allocated width, just like upstream.
        for start in tl.range(0, stride, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            tl.store(dst + offsets, 0, mask=offsets < stride)
        return

    req_idx = tl.load(mapping + row)
    count = tl.load(num_blocks_ptr + group * num_blocks_stride + req_idx)
    src = _load_ptr(src_ptrs + group, tl.int32) + req_idx * stride
    for start in tl.range(0, count, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(src + offsets, mask=offsets < count)
        # Preserve the original unwritten tail for real requests.
        tl.store(dst + offsets, values, mask=offsets < count)


def fused_gather(
    *, global_idx, local_idx, src_ptrs, global_out_ptrs, local_out_ptrs, strides, num_blocks, global_pad, local_pad
):
    groups = num_blocks.shape[0]
    if groups == 0 or global_pad + local_pad == 0:
        return
    _fused_gather_kernel[(groups, global_pad + local_pad)](
        global_idx,
        local_idx,
        src_ptrs,
        global_out_ptrs,
        local_out_ptrs,
        strides,
        num_blocks,
        num_blocks.stride(0),
        global_idx.numel(),
        local_idx.numel(),
        global_pad,
        BLOCK_SIZE=1024,
    )
