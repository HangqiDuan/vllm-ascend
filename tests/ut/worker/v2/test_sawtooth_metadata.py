# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_ascend.worker.v2.sawtooth_metadata import SawtoothSASCache


def test_interleaved_chunks_and_changed_signature_do_not_reuse_wrong_result():
    first, second = SawtoothSASCache(), SawtoothSASCache()
    calls = []

    def build(value):
        calls.append(value)
        return torch.tensor([value])

    key = ((0, 10240), (10240,), 4)
    assert first.get_or_build(key, lambda: build(11)).item() == 11
    assert second.get_or_build(key, lambda: build(22)).item() == 22
    assert first.get_or_build(key, lambda: build(99)).item() == 11
    assert first.get_or_build(((0, 5120), (10240,), 4), lambda: build(33)).item() == 33
    assert first.get_or_build(((0, 10240), (10240,), 128), lambda: build(44)).item() == 44
    assert calls == [11, 22, 33, 44]


def test_cached_result_does_not_alias_producer_scratch():
    cache = SawtoothSASCache()
    scratch = torch.tensor([7])
    cache.get_or_build("key", lambda: scratch)
    scratch.fill_(8)
    assert cache.get_or_build("key", lambda: scratch).item() == 7


def test_failed_build_is_not_cached():
    cache = SawtoothSASCache()

    def fail():
        raise RuntimeError("producer failed")

    import pytest

    with pytest.raises(RuntimeError):
        cache.get_or_build("key", fail)
    assert cache.get_or_build("key", lambda: torch.tensor([9])).item() == 9


def _decode_batch(**changes):
    from types import SimpleNamespace

    import numpy as np

    values = dict(
        is_dummy=False,
        has_prefill=False,
        num_reqs=2,
        req_ids=["a", "b"],
        num_scheduled_tokens=np.array([1, 1]),
        num_computed_tokens_np=np.array([128, 256]),
        num_reqs_after_padding=4,
        num_tokens_after_padding=4,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_decode_scope_tracks_request_order_versions_and_graph_padding():
    cache = SawtoothSASCache(enable_decode=True)
    batch = _decode_batch()
    assert cache.for_input(batch, full_graph=True) is cache
    original = cache.decode_signature
    assert cache.for_input(_decode_batch(), full_graph=True) is cache
    assert cache.decode_signature == original
    for changes in [
        dict(req_ids=["b", "a"]),
        dict(num_computed_tokens_np=batch.num_computed_tokens_np + 1),
        dict(num_tokens_after_padding=8),
    ]:
        cache.for_input(_decode_batch(**changes), full_graph=True)
        assert cache.decode_signature != original


def test_decode_scope_excludes_capture_dummy_prefill_speculation_and_device_producer():
    import numpy as np

    cache = SawtoothSASCache(enable_decode=True)
    for changes, flags in [
        (dict(is_dummy=True), {}),
        ({}, dict(for_capture=True)),
        ({}, dict(device_metadata=True)),
        ({}, dict(speculative=True)),
        (dict(has_prefill=True), {}),
        (dict(num_scheduled_tokens=np.array([1, 2])), {}),
    ]:
        assert cache.for_input(_decode_batch(**changes), full_graph=True, **flags) is None
        assert cache.decode_signature is None
    assert SawtoothSASCache().for_input(_decode_batch(), full_graph=True) is None
    assert cache.for_input(_decode_batch(has_prefill=True), full_graph=False) is cache
    assert (
        SawtoothSASCache(enable_prefill=False, enable_decode=True).for_input(
            _decode_batch(has_prefill=True), full_graph=False
        )
        is None
    )


def test_decode_scope_reuses_global_metadata_on_empty_pcp_shard():
    import numpy as np

    cache = SawtoothSASCache(enable_decode=True)
    global_batch = _decode_batch()
    empty = _decode_batch(
        num_reqs=0,
        req_ids=[],
        num_scheduled_tokens=np.array([], dtype=int),
        num_computed_tokens_np=np.array([], dtype=int),
        num_reqs_after_padding=1,
        num_tokens_after_padding=1,
    )
    assert cache.for_input(empty, full_graph=True, global_batch=global_batch) is cache
    signature = cache.decode_signature
    assert cache.for_input(empty, full_graph=True, global_batch=_decode_batch()) is cache
    assert cache.decode_signature == signature
    cache.for_input(
        empty,
        full_graph=True,
        global_batch=_decode_batch(
            num_computed_tokens_np=global_batch.num_computed_tokens_np + 1,
        ),
    )
    assert cache.decode_signature != signature
    empty.num_tokens_after_padding = 2
    cache.for_input(empty, full_graph=True, global_batch=global_batch)
    assert cache.decode_signature != signature


def test_global_decode_scope_keeps_capture_dummy_and_prefill_guards():
    cache = SawtoothSASCache(enable_decode=True)
    for local, global_batch, flags in [
        (_decode_batch(is_dummy=True), _decode_batch(), {}),
        (_decode_batch(), _decode_batch(is_dummy=True), {}),
        (_decode_batch(), _decode_batch(has_prefill=True), {}),
        (_decode_batch(), _decode_batch(), dict(for_capture=True)),
        (_decode_batch(), _decode_batch(), dict(device_metadata=True)),
        (_decode_batch(), _decode_batch(), dict(speculative=True)),
    ]:
        assert cache.for_input(local, full_graph=True, global_batch=global_batch, **flags) is None
        assert cache.decode_signature is None
