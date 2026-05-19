# ruff: noqa: F401
import asyncio
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time
import threading
import sys
import types

import pytest

pytest.importorskip("ray")

import ray

from mint_server.backend.megatron_distributed import MegatronSessionStateManager, MegatronWorkerGroup
from mint_server.backend.training_session_manager import TrainingSession
from mint_server.backend.verl_training import TrainingWorker, VerlTrainingEngine

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
    "_megatron_load_meta",
    "_completed_ray_ref",
    "_failed_ray_ref",
    "_labeled_ray_ref",
]


class _RecordingRemoteMethod:
    def __init__(self, ref: object):
        self._ref = ref
        self.calls: list[tuple[tuple, dict]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _coerce_ray_ref(self._ref)


_UNSET = object()


class _FakeRayRef:
    def __init__(
        self,
        label: object | None = None,
        *,
        result: object = _UNSET,
        exc: BaseException | None = None,
    ):
        self._label = label
        self._result = result
        self._exc = exc
        self._future: concurrent.futures.Future | None = None

    def __eq__(self, other):
        return self._label == other

    def __hash__(self):
        return hash(self._label)

    def __repr__(self):
        return repr(self._label)

    def future(self):
        if self._future is not None:
            return self._future
        future: concurrent.futures.Future = concurrent.futures.Future()
        if self._exc is not None:
            future.set_exception(self._exc)
        elif self._result is not _UNSET:
            future.set_result(self._result)
        else:
            def _resolve():
                try:
                    result = ray.get(self._label, timeout=1.0)
                    if not future.cancelled():
                        future.set_result(result)
                except BaseException as exc:
                    if not future.cancelled():
                        future.set_exception(exc)

            threading.Thread(target=_resolve, daemon=True).start()
        self._future = future
        return future


def _coerce_ray_ref(ref: object) -> object:
    if isinstance(ref, str):
        return _FakeRayRef(ref)
    return ref


def _completed_ray_ref(result: object = None) -> _FakeRayRef:
    return _FakeRayRef(result=result)


def _failed_ray_ref(exc: BaseException) -> _FakeRayRef:
    return _FakeRayRef(exc=exc)


def _labeled_ray_ref(label: object) -> _FakeRayRef:
    return _FakeRayRef(label)


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
        self.mark_session_loaded = _RecordingRemoteMethod(_completed_ray_ref({"status": "ok"}))


async def _noop_log_worker_request_context(*args, **kwargs):
    return None


def _megatron_load_meta(
    *,
    current_step: int,
    learning_rate: float,
    actual_rank: int,
    checkpoint_path: str,
    optimizer_restored: bool = True,
    actor_only_state_dirty: bool | None = None,
    train_attn: bool = True,
    train_mlp: bool = True,
    train_unembed: bool = True,
) -> dict:
    return {
        "current_step": current_step,
        "learning_rate": learning_rate,
        "actual_rank": actual_rank,
        "actor_only_state_dirty": optimizer_restored if actor_only_state_dirty is None else actor_only_state_dirty,
        "checkpoint_path": checkpoint_path,
        "optimizer_restored": optimizer_restored,
        "train_attn": train_attn,
        "train_mlp": train_mlp,
        "train_unembed": train_unembed,
    }
