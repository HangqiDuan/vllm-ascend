# SPDX-License-Identifier: Apache-2.0
"""Immutable SAS operator results owned by one admitted Sawtooth chunk."""

from collections.abc import Callable, Hashable

import torch


class SawtoothSASCache:
    def __init__(self, *, enable_prefill: bool = True, enable_decode: bool = False):
        self._entries = {}
        self.enable_prefill = enable_prefill
        self.enable_decode = enable_decode
        self.decode_signature = None

    def for_input(
        self,
        input_batch,
        *,
        full_graph: bool,
        global_batch=None,
        for_capture: bool = False,
        device_metadata: bool = False,
        speculative: bool = False,
    ):
        """Select host-side reuse without reading any device tensors.

        The owner already keys this cache by immutable request steps. Include
        row order and graph padding as well: equal shapes alone are not enough.
        Graph capture and device-side metadata producers retain their own path.
        """
        self.decode_signature = None
        logical_batch = global_batch if global_batch is not None else input_batch
        if for_capture or input_batch.is_dummy or logical_batch.is_dummy:
            return None
        if logical_batch.has_prefill:
            return self if self.enable_prefill and not full_graph else None
        if not self.enable_decode or device_metadata or speculative:
            return None
        # An empty PCP shard still builds metadata for the global Decode batch.
        num_reqs = logical_batch.num_reqs
        if not num_reqs or not (logical_batch.num_scheduled_tokens[:num_reqs] == 1).all():
            return None
        self.decode_signature = (
            tuple(logical_batch.req_ids[:num_reqs]),
            tuple(logical_batch.num_computed_tokens_np[:num_reqs].tolist()),
            logical_batch.num_reqs_after_padding,
            logical_batch.num_tokens_after_padding,
            input_batch.num_reqs_after_padding,
            input_batch.num_tokens_after_padding,
        )
        return self

    def get_or_build(self, key: Hashable, build: Callable[[], torch.Tensor]) -> torch.Tensor:
        entry = self._entries.get(key)
        if entry is None:
            # Never retain a builder's reusable scratch/output buffer.
            value = build().clone()
            event = None
            if value.device.type == "npu":
                event = torch.npu.Event()
                event.record(torch.npu.current_stream())
            self._entries[key] = value, event
        else:
            value, event = entry
            if event is not None:
                stream = torch.npu.current_stream()
                stream.wait_event(event)
                # The chunk can retire on the host before this consumer finishes.
                value.record_stream(stream)
        return value
