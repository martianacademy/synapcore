#!/usr/bin/env python3
"""
SynapCore Modular Thalamic Router Package (`native/router/__init__.py`)
=======================================================================
WHAT THIS PACKAGE DOES:
  Organizes the Cortical-Thalamic Router v2 into clean, self-contained, importable
  neuroscience-inspired components so each biological mechanism can be understood,
  tested, and ablated independently:

  - `native/router/config.py`         : `ThalamicRouterV2Config` & Telemetry/Ablation TypedDicts
  - `native/router/inertia.py`        : `NeuralInertiaWorkingMemory` (Working-Memory EMA `s_t`)
  - `native/router/topdown.py`        : `TopDownThalamocorticalLoop` (Layer-15 -> Layers 0..15 `c_{t-1}`)
  - `native/router/neuromodulator.py` : `HomeostaticNeuromodulator` (Z-Score Entropy Gate & Top-8 Mass Conservation)
  - `native/router/hebbian.py`        : `HebbianSynapticPlasticity` (Symmetric Oja `256x256` Co-Activation & Fatigue)
  - `native/router/thalamic_v2.py`    : `ThalamicRouterV2` Main Coordinator, True Hidden-State SFT Trainer & Ablation Runner
"""

from __future__ import annotations

from native.router.config import (
    SFTBatchTrainingReport,
    StageAblationEntry,
    StageBiasDiagnostics,
    ThalamicRouterV2Config,
    ThalamicV2Telemetry,
)
from native.router.hebbian import HebbianSynapticPlasticity
from native.router.inertia import NeuralInertiaWorkingMemory
from native.router.neuromodulator import HomeostaticNeuromodulator
from native.router.thalamic_v2 import ThalamicRouterV2
from native.router.topdown import TopDownThalamocorticalLoop

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
