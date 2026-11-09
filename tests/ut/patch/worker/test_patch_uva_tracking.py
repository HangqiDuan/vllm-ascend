# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.patch.worker.patch_uva import UvaBufferWrapper


def cpu_buffer(shape):
    # Exercise the real tracking/copy logic with CPU storage; hardware transfer
    # is checked separately on NPU. Avoid allocating a device in unit tests.
    shape = (shape,) if isinstance(shape, int) else shape
    buf = object.__new__(UvaBufferWrapper)
    buf._cpu = torch.zeros(shape, dtype=torch.int32)
    buf._np = buf._cpu.numpy()
    buf._uva = torch.full(shape, -1, dtype=torch.int32)
    buf._use_real_uva = False
    buf._modified_indices = set()
    buf._dirty_prefix_end = 0
    return buf


@pytest.mark.parametrize("view", ["np", "cpu"])
def test_long_prefix_does_not_visit_every_row(view):
    buf = cpu_buffer(131160)
    values = np.arange(131155, dtype=np.int32)
    if view == "cpu":
        values = torch.from_numpy(values)
    # Regression: tracking a contiguous prompt must take constant bookkeeping
    # work, instead of making one Python callback per token.
    with patch.object(
        buf, "_mark_cpu_modified", side_effect=AssertionError("per-row callback")
    ):
        getattr(buf, view)[:131155] = values
    actual = buf.uva()
    assert torch.equal(actual[:131155], torch.arange(131155, dtype=torch.int32))
    assert (actual[131155:] == -1).all()
    assert not buf._modified_indices and buf._dirty_prefix_end == 0
    # A second read without a tracked write must not re-copy the host buffer.
    buf._cpu[0] = -7
    assert buf.uva()[0] == 0


@pytest.mark.parametrize("view", ["np", "cpu"])
def test_overlapping_prefix_and_scalar_writes(view):
    buf = cpu_buffer((8, 3))
    dst = getattr(buf, view)
    dst[:3] = 2
    dst[:5] = 4
    dst[2] = 8
    dst[5] = 9
    expected = torch.full((8, 3), -1, dtype=torch.int32)
    expected[:5] = 4
    expected[2] = 8
    expected[5] = 9
    assert torch.equal(buf.uva(), expected)


@pytest.mark.parametrize("view", ["np", "cpu"])
def test_prefix_and_noncontiguous_rows_preserve_clean_rows(view):
    buf = cpu_buffer((10, 2))
    dst = getattr(buf, view)
    dst[:2] = 3
    dst[4:8:2] = 5
    dst[-1] = 9
    expected = torch.full((10, 2), -1, dtype=torch.int32)
    expected[:2] = 3
    expected[4:8:2] = 5
    expected[-1] = 9
    original_to = torch.Tensor.to

    def to_cpu(self, *args, **kwargs):
        if kwargs.get("device") == "npu":
            assert kwargs.get("non_blocking") is True
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    with (
        patch.object(torch.Tensor, "pin_memory", lambda self: self),
        patch.object(torch.Tensor, "to", to_cpu),
    ):
        assert torch.equal(buf.uva(), expected)
    assert not buf._modified_indices and buf._dirty_prefix_end == 0


def test_failed_copy_keeps_dirty_state_for_retry():
    buf = cpu_buffer((8, 2))
    buf.np[:4] = 7
    buf.np[4] = 9
    with (
        patch.object(torch.Tensor, "copy_", side_effect=RuntimeError("copy failed")),
        pytest.raises(RuntimeError, match="copy failed"),
    ):
        buf.uva()
    assert buf._dirty_prefix_end == 4
    assert buf._modified_indices == {4}
    expected = torch.full((8, 2), -1, dtype=torch.int32)
    expected[:4] = 7
    expected[4] = 9
    assert torch.equal(buf.uva(), expected)
    assert not buf._modified_indices and buf._dirty_prefix_end == 0


@pytest.mark.parametrize("view", ["np", "cpu"])
def test_empty_partial_column_and_failed_writes(view):
    buf = cpu_buffer((6, 3))
    dst = getattr(buf, view)
    dst[:0] = 1
    assert (buf.uva() == -1).all()
    dst[:4, 1] = 7
    actual = buf.uva()
    assert (actual[:4, 1] == 7).all()
    assert (actual[:4, 0] == 0).all()
    assert (actual[4:] == -1).all()
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        dst[100] = 2
    assert not buf._modified_indices and buf._dirty_prefix_end == 0
