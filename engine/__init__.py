"""
SynapCore Multi-Request Continuous Batching Scheduler (`engine/scheduler.py`)
=============================================================================
Implements Task P2.3:
  - `ContinuousBatchScheduler`: Dynamic token batching across concurrent requests with
    per-request Paged KV Cache (`PagedKVArena` + request KV state) and a single shared
    `HotExpertLFULRUCache` + grouped active expert execution across the batch.
"""

from engine.scheduler import (
    BatchRequestState,
    BatchSchedulerMetrics,
    BatchStepResult,
    ContinuousBatchScheduler,
    benchmark_continuous_batch_scheduler,
)

__all__ = [
    "BatchRequestState",
    "BatchSchedulerMetrics",
    "BatchStepResult",
    "ContinuousBatchScheduler",
    "benchmark_continuous_batch_scheduler",
]
