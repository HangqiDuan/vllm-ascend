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
    cache.get_or_build('key', lambda: scratch)
    scratch.fill_(8)
    assert cache.get_or_build('key', lambda: scratch).item() == 7


def test_failed_build_is_not_cached():
    cache = SawtoothSASCache()
    def fail():
        raise RuntimeError('producer failed')
    import pytest
    with pytest.raises(RuntimeError):
        cache.get_or_build('key', fail)
    assert cache.get_or_build('key', lambda: torch.tensor([9])).item() == 9
