"""Minimal per-iteration latency and CUDA-memory measurement for evaluation."""

import json
import os
import time

import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


def _rank_path(path, rank):
    stem, extension = os.path.splitext(path)
    extension = extension or '.jsonl'
    return f'{stem}.rank{int(rank)}{extension}'


@HOOKS.register_module()
class InferenceBenchmarkHook(Hook):
    """Measure a real isolated deployment profile, not a shared screen."""

    priority = 'VERY_HIGH'

    def __init__(self, output_path, profile, dataset, warmup=2):
        self.output_path = str(output_path)
        self.profile = str(profile)
        self.dataset = str(dataset)
        self.warmup = max(0, int(warmup))
        self._start = None

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._sync()
        self._start = time.perf_counter()

    def after_test_iter(
            self, runner, batch_idx, data_batch=None, outputs=None):
        self._sync()
        latency = time.perf_counter() - self._start
        peak_allocated = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else 0)
        peak_reserved = (
            int(torch.cuda.max_memory_reserved())
            if torch.cuda.is_available() else 0)
        model = runner.model.module if hasattr(
            runner.model, 'module') else runner.model
        cost = dict(getattr(model, '_inference_profile_cost', {}) or {})
        record = dict(
            schema_version=1,
            rank=int(runner.rank),
            batch_index=int(batch_idx),
            dataset=self.dataset,
            profile=self.profile,
            warmup=bool(batch_idx < self.warmup),
            latency_seconds=float(latency),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            **cost,
        )
        path = _rank_path(self.output_path, runner.rank)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
