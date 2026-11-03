#!/usr/bin/env python3
"""
SynapCore Thalamic Router v2 — Backwards-Compatible Entry Point (`native/thalamic_router_v2.py`)
================================================================================================
WHAT THIS FILE DOES:
  Re-exports all modular components from the `native/router/` package:
    - `native/router/config.py`         : `ThalamicRouterV2Config`, `ThalamicV2Telemetry`, `StageBiasDiagnostics`
    - `native/router/inertia.py`        : `NeuralInertiaWorkingMemory` (Working-Memory EMA `s_t`)
    - `native/router/topdown.py`        : `TopDownThalamocorticalLoop` (Layer-15 Feedback `c_{t-1}`)
    - `native/router/neuromodulator.py` : `HomeostaticNeuromodulator` (Z-Score Entropy Gate & Mass Conservation)
    - `native/router/hebbian.py`        : `HebbianSynapticPlasticity` (Symmetric Oja `256x256` Co-Activation)
    - `native/router/thalamic_v2.py`    : `ThalamicRouterV2` Coordinator & True Layer-Hidden-State SFT Trainer

  Keeping this module as a clean facade ensures that any existing code importing
  `from native.thalamic_router_v2 import ThalamicRouterV2` continues to work seamlessly
  while all implementation logic lives in dedicated, easy-to-read files under `native/router/`.
"""

from __future__ import annotations

from native.router import (
    HebbianSynapticPlasticity,
    HomeostaticNeuromodulator,
    NeuralInertiaWorkingMemory,
    SFTBatchTrainingReport,
    StageAblationEntry,
    StageBiasDiagnostics,
    ThalamicRouterV2,
    ThalamicRouterV2Config,
    ThalamicV2Telemetry,
    TopDownThalamocorticalLoop,
)

__all__ = [
    "HebbianSynapticPlasticity",
    "HomeostaticNeuromodulator",
    "NeuralInertiaWorkingMemory",
    "SFTBatchTrainingReport",
    "StageAblationEntry",
    "StageBiasDiagnostics",
    "ThalamicRouterV2",
    "ThalamicRouterV2Config",
    "ThalamicV2Telemetry",
    "TopDownThalamocorticalLoop",
]
