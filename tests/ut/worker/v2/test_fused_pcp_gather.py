from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm_ascend.worker.v2.pcp_manager as module


def setup_manager(monkeypatch, shared=None):
    manager = module.AscendPCPManager.__new__(module.AscendPCPManager)
    manager._fused_pcp_gather_enabled = True
    manager._prepared_global_gather = None
    manager._global_batch = SimpleNamespace(
        idx_mapping=torch.tensor([2, 0]), num_reqs_after_padding=2,
        has_prefill=False, is_dummy=False, num_tokens_after_padding=2,
        is_prefilling_np=np.array([False, False]),
    )
    manager._local_batch = SimpleNamespace(
        idx_mapping=torch.tensor([0]), num_reqs_after_padding=1,
        is_dummy=False, is_cpp_profile=False, num_tokens=1,
    )
    manager._local_block_tables = (torch.full((4, 4), -1),)
    manager._local_block_table_ptrs = torch.tensor([1])
    manager._block_tables = shared or SimpleNamespace(
        num_kv_cache_groups=1, block_table_ptrs=torch.tensor([2]),
        block_table_strides=torch.tensor([4]),
        num_blocks=SimpleNamespace(gpu=torch.tensor([[4, 4, 4]])),
        input_block_tables=[torch.full((4, 4), -2)],
        input_block_table_ptrs=torch.tensor([3]),
        gather_block_tables=Mock(return_value=(torch.tensor([[77]]),)),
    )
    manager._global_batch_slot_mappings = torch.zeros((1, 8))
    manager._hidden_restore_idx = torch.tensor([0, 1])
    manager._local_gather_idx = torch.tensor([0])
    manager._padded_gather_idx = None
    manager._gathered_kv_write_mask = None
    manager.dcp_world_size = 1
    manager.prepare_slot_mappings = Mock(return_value=torch.tensor([[9]]))
    fusion = Mock()
    monkeypatch.setattr(module, 'fused_gather', fusion, raising=False)
    fallback = Mock(return_value=((torch.tensor([[88]]),), torch.tensor([[99]])))
    monkeypatch.setattr(module.PCPManager, 'prepare_attn', fallback)
    return manager, fusion, fallback


def test_prepares_both_views_and_consumes_global_once(monkeypatch):
    m, fusion, fallback = setup_manager(monkeypatch)
    local, slots = m.prepare_attn(m._local_batch)
    fusion.assert_called_once()
    fallback.assert_not_called()
    args = fusion.call_args.kwargs
    assert args['global_idx'] is m._global_batch.idx_mapping
    assert args['local_idx'] is m._local_batch.idx_mapping
    assert args['global_out_ptrs'] is m._block_tables.input_block_table_ptrs
    assert args['local_out_ptrs'] is m._local_block_table_ptrs
    assert local[0].data_ptr() == m._local_block_tables[0].data_ptr()
    assert local[0].shape == (1, 4)
    assert slots is m.prepare_slot_mappings.return_value
    context = m.build_attention_context(m._local_batch, local, slots)
    assert context.global_block_tables[0].data_ptr() == m._block_tables.input_block_tables[0].data_ptr()
    assert context.global_block_tables[0].shape == (2, 4)
    m._block_tables.gather_block_tables.assert_not_called()
    m.build_attention_context(m._local_batch, local, slots)
    m._block_tables.gather_block_tables.assert_called_once()


@pytest.mark.parametrize('change', ['disabled', 'prefill', 'dummy', 'no_global'])
def test_ineligible_batches_use_original_path(monkeypatch, change):
    m, fusion, fallback = setup_manager(monkeypatch)
    if change == 'disabled':
        m._fused_pcp_gather_enabled = False
    elif change == 'prefill':
        m._global_batch.has_prefill = True
    elif change == 'dummy':
        m._local_batch.is_dummy = True
    else:
        m._global_batch = None
    assert m.prepare_attn(m._local_batch) is fallback.return_value
    fusion.assert_not_called()
    assert m._prepared_global_gather is None


@pytest.mark.parametrize('change', ['local', 'global', 'other_manager', 'failure'])
def test_stale_or_failed_preparation_does_not_reuse_global(monkeypatch, change):
    m, fusion, _ = setup_manager(monkeypatch)
    local, slots = m.prepare_attn(m._local_batch)
    if change == 'local':
        m._local_batch = SimpleNamespace(**vars(m._local_batch))
    elif change == 'global':
        m._global_batch = SimpleNamespace(**vars(m._global_batch))
    elif change == 'other_manager':
        peer, _, _ = setup_manager(monkeypatch, m._block_tables)
        peer.prepare_attn(peer._local_batch)
    else:
        m.prepare_slot_mappings.side_effect = RuntimeError('slot failure')
        with pytest.raises(RuntimeError, match='slot failure'):
            m.prepare_attn(m._local_batch)
    m.build_attention_context(m._local_batch, local, slots)
    m._block_tables.gather_block_tables.assert_called_once()


def test_empty_local_shard_still_prepares_nonempty_global(monkeypatch):
    m, fusion, _ = setup_manager(monkeypatch)
    m._local_batch.idx_mapping = torch.empty(0, dtype=torch.int64)
    m._local_batch.num_tokens = 0
    m.prepare_attn(m._local_batch)
    fusion.assert_called_once()
    assert fusion.call_args.kwargs['local_idx'].numel() == 0
    assert fusion.call_args.kwargs['local_pad'] == 1


def test_dummy_context_discards_pending_global(monkeypatch):
    m, _, _ = setup_manager(monkeypatch)
    local, slots = m.prepare_attn(m._local_batch)
    dummy = SimpleNamespace(is_dummy=True, is_cpp_profile=False,
                            num_tokens_after_padding=1, num_reqs_after_padding=1, num_reqs=1)
    monkeypatch.setattr(module.AscendPCPManager, 'is_decode_sharded', property(lambda _: False))
    m.pcp_rank = 0
    m.pcp_world_size = 1
    m.device = 'cpu'
    m.build_attention_context(dummy, local, slots)
    m.build_attention_context(m._local_batch, local, slots)
    m._block_tables.gather_block_tables.assert_called_once()
