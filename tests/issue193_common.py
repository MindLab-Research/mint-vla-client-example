# ruff: noqa: F401
import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time
import sys
import types

import pytest

pytest.importorskip("ray")

import ray

from tinker_server.backend.megatron_distributed import MegatronSessionStateManager, MegatronWorkerGroup
from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.backend.verl_training import TrainingWorker, VerlTrainingEngine

__all__ = [
    "asyncio",
    "json",
    "logging",
    "os",
    "Path",
    "SimpleNamespace",
    "time",
    "sys",
    "types",
    "pytest",
    "ray",
    "MegatronSessionStateManager",
    "MegatronWorkerGroup",
    "TrainingSession",
    "TrainingWorker",
    "VerlTrainingEngine",
    "_RecordingRemoteMethod",
    "_FakeWorker",
    "_FakeSamplerWorker",
    "_FakeLoadWorker",
    "_noop_log_worker_request_context",
]


class _RecordingRemoteMethod:
    def __init__(self, ref: str):
        self._ref = ref
        self.calls: list[tuple[tuple, dict]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._ref


class _HeartbeatWorkerMixin:
    def __init__(self):
        self.heartbeat = _RecordingRemoteMethod("heartbeat-ref")


class _FakeWorker(_HeartbeatWorkerMixin):
    def __init__(self, ref: str = "fake-save-checkpoint-ref"):
        super().__init__()
        self.save_checkpoint = _RecordingRemoteMethod(ref)


class _FakeSamplerWorker(_HeartbeatWorkerMixin):
    def __init__(self, ref: str = "fake-save-lora-ref"):
        super().__init__()
        self.save_lora_weights = _RecordingRemoteMethod(ref)


class _FakeLoadWorker(_HeartbeatWorkerMixin):
    def __init__(self, ref: str = "fake-load-checkpoint-ref"):
        super().__init__()
        self.__ray_ready__ = _RecordingRemoteMethod("fake-load-ready-ref")
        self.load_checkpoint = _RecordingRemoteMethod(ref)
        self.mark_session_loaded = _RecordingRemoteMethod("fake-mark-session-loaded-ref")


async def _noop_log_worker_request_context(*args, **kwargs):
    return None
