"""Protocol tests prepared locally; run against the isolated candidate before edits."""
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils


class Stream:
    def __init__(self):
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)

    def record_event(self):
        return SimpleNamespace(query=lambda: True)

    def wait_event(self, event):
        self.waits.append(event)


def batch(rows=2, mixed=False, no_samples=False):
    computed = np.full(rows, 100, dtype=np.int32)
    prompt = np.full(rows, 50, dtype=np.int32)
    if mixed:
        prompt[0] = 1000
    if no_samples:
        prompt[:] = 1000
    mapping = np.arange(rows, dtype=np.int32)
    return SimpleNamespace(
        num_reqs=rows, num_computed_tokens_np=computed, prefill_len_np=prompt,
        num_scheduled_tokens=np.ones(rows, dtype=np.int32),
        is_prefilling_np=prompt > computed, max_seq_len_np=None,
        idx_mapping_np=mapping, idx_mapping=torch.from_numpy(mapping.copy()),
    )


@pytest.fixture
def transport(monkeypatch):
    wire = deque()
    sent = []
    receiving = [False]
    recorded = []
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: Stream())
    monkeypatch.setattr(torch.cuda, 'Stream', lambda _: Stream())
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda t, s: recorded.append((t, s)))
    monkeypatch.setattr(pp_utils.current_platform, 'is_xpu', lambda: False)
    monkeypatch.setattr(pp_utils, 'async_tensor_h2d', lambda data, device: torch.tensor(data))
    group = SimpleNamespace(is_last_rank=True, last_rank=1, world_size=2,
                            make_sibling_device_group=lambda **_: object())
    monkeypatch.setattr(pp_utils, 'get_pp_group', lambda: group)

    def broadcast(tensor, *, src, group):
        assert src == 1
        if receiving[0]:
            payload = wire.popleft()
            assert tensor.dtype == payload.dtype and tensor.shape == payload.shape
            tensor.copy_(payload)
        else:
            sent.append(tensor.clone())
            wire.append(tensor.clone())

    monkeypatch.setattr(torch.distributed, 'broadcast', broadcast)

    def handlers(enabled=True, speculative=0):
        monkeypatch.setenv('VLLM_ASCEND_PACKED_PP_SAMPLES', '1' if enabled else '0')
        group.is_last_rank = True
        sender = pp_utils.PPHandler(8, speculative, torch.device('cpu'))
        group.is_last_rank = False
        receiver = pp_utils.PPHandler(8, speculative, torch.device('cpu'))
        receiver.queue = deque([None])
        return sender, receiver

    return SimpleNamespace(wire=wire, sent=sent, receiving=receiving,
                           recorded=recorded, handlers=handlers)


@pytest.mark.parametrize('rows', [1, 2, 4])
@pytest.mark.parametrize('enabled', [False, True])
def test_exact_bits_and_collective_count(transport, rows, enabled):
    sender, receiver = transport.handlers(enabled)
    data = batch(rows)
    # Values beyond int32 catch accidental narrowing; strided input catches
    # assumptions about sampler tensor contiguity.
    tokens = (torch.arange(rows * 2, dtype=torch.int64) + 2**40).view(rows, 2)[:, :1]
    sampled = torch.arange(rows, dtype=torch.int32)
    rejected = -torch.arange(rows, dtype=torch.int32) - 1
    sender.broadcast(tokens, sampled, rejected, data)
    assert len(transport.sent) == (1 if enabled else 2)
    transport.receiving[0] = True
    assert receiver.receive(data)
    pending = receiver.queue[-1]
    assert pending.sampled_tokens.dtype == torch.int64
    assert pending.num_sampled.dtype == pending.num_rejected.dtype == torch.int32
    assert torch.equal(pending.sampled_tokens, tokens)
    assert torch.equal(pending.num_sampled, sampled)
    assert torch.equal(pending.num_rejected, rejected)
    assert not transport.wire
    assert transport.recorded


def test_mixed_batch_and_freed_generation_are_masked_at_consumption(transport):
    sender, receiver = transport.handlers()
    data = batch(4, mixed=True)
    tokens = torch.arange(4, dtype=torch.int64).view(4, 1)
    sampled, rejected = torch.ones(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32)
    sender.broadcast(tokens, sampled, rejected, data)
    transport.receiving[0] = True
    assert not receiver.receive(data)
    pending = receiver.queue[-1]
    assert pending.need_sampled_mask.tolist() == [False, True, True, True]
    receiver.on_req_idx_freed(1)
    output = receiver.get_prev_sampled_outputs()
    assert output['idx_mapping'].tolist() == [-1, -1, 2, 3]
    assert torch.equal(output['sampled_tokens'], tokens)
    assert pending.event in receiver.main_stream.waits


def test_pending_payload_is_not_overwritten_by_next_receive(transport):
    sender, receiver = transport.handlers()
    data = batch()
    counts = torch.tensor([1, 0], dtype=torch.int32)
    rejected = torch.tensor([0, 1], dtype=torch.int32)
    first = torch.tensor([[11], [12]], dtype=torch.int64)
    sender.broadcast(first, counts, rejected, data)
    transport.receiving[0] = True
    receiver.receive(data)
    pending = receiver.queue[-1]
    transport.receiving[0] = False
    second = torch.tensor([[21], [22]], dtype=torch.int64)
    sender.broadcast(second, counts, rejected, data)
    transport.receiving[0] = True
    receiver.receive(data)
    assert torch.equal(pending.sampled_tokens, first)
    assert torch.equal(receiver.queue[-1].sampled_tokens, second)
    assert pending.sampled_tokens.data_ptr() != receiver.queue[-1].sampled_tokens.data_ptr()


def test_no_sample_batch_skips_all_collectives(transport):
    sender, receiver = transport.handlers()
    data = batch(no_samples=True)
    sender.broadcast(torch.zeros(2, 1, dtype=torch.int64),
                     torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), data)
    transport.receiving[0] = True
    assert not receiver.receive(data)
    assert not transport.sent and not transport.wire
    assert receiver.queue[-1] is None


def test_speculative_mode_keeps_original_three_message_receive(transport):
    sender, receiver = transport.handlers(speculative=2)
    data = batch()
    tokens = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.int64)
    counts = torch.tensor([3, 1], dtype=torch.int32)
    rejected = torch.tensor([0, 2], dtype=torch.int32)
    sender.broadcast(tokens, counts, rejected, data)
    assert len(transport.sent) == 2
    drafts = torch.tensor([[20, 21], [22, 23]], dtype=torch.int64)
    transport.wire.append(drafts)
    transport.receiving[0] = True
    receiver.receive(data)
    pending = receiver.queue[-1]
    assert torch.equal(pending.sampled_tokens, tokens)
    assert torch.equal(pending.draft_tokens, drafts)
    assert not transport.wire
