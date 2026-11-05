# SPDX-License-Identifier: Apache-2.0
"""Immutable SAS operator results owned by one admitted Sawtooth chunk."""

from collections.abc import Callable, Hashable

import torch


class SawtoothSASCache:
    def __init__(self):
        self._entries = {}

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
