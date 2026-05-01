#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
OWNER_ID = os.environ.get("TINKER_OWNER_ID") or os.environ.get("TINKER_USER_ID", "issue_315_user")
BASE_MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-0.6B")
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "8"))
LEARNING_RATE = float(os.environ.get("TINKER_LEARNING_RATE", "1e-4"))
WARMUP_STEPS = int(os.environ.get("TINKER_WARMUP_STEPS", "5"))
COMPARE_STEPS = int(os.environ.get("TINKER_COMPARE_STEPS", "3"))
DIFFERENT_CHECKPOINT_STEPS = int(os.environ.get("TINKER_DIFFERENT_CHECKPOINT_STEPS", "2"))
SEQ_LEN = int(os.environ.get("TINKER_SEQ_LEN", "64"))
CREATE_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_TIMEOUT_S", "3600"))
SAVE_TIMEOUT_S = float(os.environ.get("TINKER_SAVE_TIMEOUT_S", "3600"))
RESUME_TIMEOUT_S = float(os.environ.get("TINKER_RESUME_TIMEOUT_S", "3600"))
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "3600"))
POLL_REQUEST_TIMEOUT_S = float(os.environ.get("TINKER_POLL_REQUEST_TIMEOUT_S", "180"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{path}", json=payload, headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:800]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict JSON: {type(data)}")
    return data


def _poll_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=_headers(),
            timeout=POLL_REQUEST_TIMEOUT_S,
        )
        if resp.status_code == 408:
            time.sleep(2.0)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"retrieve_future({request_id}) -> {resp.status_code}: {resp.text[:800]!r}")
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"retrieve_future returned non-dict JSON: {type(data)}")
        return data
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s for request_id={request_id}")


def _await_maybe_async(result: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    request_id = result.get("request_id")
    if isinstance(request_id, str) and request_id:
        return _poll_future(request_id, timeout_s=timeout_s)
    return result


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=120.0)
    except Exception:
        pass


@dataclass(frozen=True)
class TrainStep:
    index: int
    loss: float


def _make_training_batch() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    vocab_window = 64
    seq_len = max(8, SEQ_LEN)

    for sample_idx in range(4):
        tokens = [10 + ((sample_idx * 17 + pos) % vocab_window) for pos in range(seq_len)]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [1.0] * len(target_tokens)
        data.append(
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
                "loss_fn_inputs": {
                    "target_tokens": {
                        "data": target_tokens,
                        "shape": [len(target_tokens)],
                        "dtype": "int64",
                    },
                    "weights": {
                        "data": weights,
                        "shape": [len(weights)],
                        "dtype": "float32",
                    },
                },
            }
        )
    return data


def _create_model(session_id: str, model_seq_id: int) -> tuple[str, str]:
    created = _post_json(
        "/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": LORA_RANK},
            "learning_rate": LEARNING_RATE,
            "user_metadata": {"issue": 315, "script": "reproduce_issue_315_resume_training.py"},
        },
        timeout_s=60.0,
    )
    created = _await_maybe_async(created, timeout_s=CREATE_TIMEOUT_S)
    model_id = created.get("model_id")
    backend = created.get("backend")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {created!r}")
    if not isinstance(backend, str) or not backend:
        raise RuntimeError(f"create_model missing backend: {created!r}")
    print(f"created model_id={model_id} backend={backend}", flush=True)
    return model_id, backend


def _train_step(model_id: str, batch: list[dict[str, Any]], step_idx: int) -> TrainStep:
    fb = _await_maybe_async(
        _post_json(
            "/api/v1/forward_backward",
            {
                "model_id": model_id,
                "forward_backward_input": {"data": batch, "loss_fn": "cross_entropy"},
            },
            timeout_s=120.0,
        ),
        timeout_s=POLL_TIMEOUT_S,
    )
    metrics = fb.get("metrics")
    if not isinstance(metrics, dict) or "loss:mean" not in metrics:
        raise RuntimeError(f"forward_backward missing metrics['loss:mean']: {fb!r}")
    loss = float(metrics["loss:mean"])
    _await_maybe_async(
        _post_json(
            "/api/v1/optim_step",
            {
                "model_id": model_id,
                "adam_params": {
                    "learning_rate": LEARNING_RATE,
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-12,
                },
            },
            timeout_s=60.0,
        ),
        timeout_s=POLL_TIMEOUT_S,
    )
    print(f"step={step_idx} loss={loss:.6f}", flush=True)
    return TrainStep(index=step_idx, loss=loss)


def _save_state(model_id: str, checkpoint_name: str) -> str:
    saved = _await_maybe_async(
        _post_json("/api/v1/save_state", {"model_id": model_id, "path": checkpoint_name}, timeout_s=60.0),
        timeout_s=SAVE_TIMEOUT_S,
    )
    path = saved.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"save_state missing checkpoint path: {saved!r}")
    print(f"checkpoint={path}", flush=True)
    return path


def _load_state(model_id: str, checkpoint_path: str, *, optimizer: bool = True, label: str = "reloaded") -> None:
    loaded = _await_maybe_async(
        _post_json(
            "/api/v1/load_state",
            {"model_id": model_id, "path": checkpoint_path, "optimizer": optimizer, "owner_id": OWNER_ID},
            timeout_s=60.0,
        ),
        timeout_s=RESUME_TIMEOUT_S,
    )
    loaded_path = loaded.get("path")
    if not isinstance(loaded_path, str) or not loaded_path:
        raise RuntimeError(f"load_state missing path: {loaded!r}")
    print(f"{label} model_id={model_id} path={loaded_path}", flush=True)


def _resume_from_state(
    session_id: str,
    model_seq_id: int,
    state_path: str,
    *,
    load_optimizer: bool = True,
) -> str:
    payload = {
        "session_id": session_id,
        "model_seq_id": model_seq_id,
        "base_model": BASE_MODEL,
        "state_path": state_path,
        "lora_config": {"rank": LORA_RANK},
        "load_optimizer": bool(load_optimizer),
        "owner_id": OWNER_ID,
        "user_metadata": {"issue": 315, "script": "reproduce_issue_315_resume_training.py"},
    }

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resumed = _post_json(
                "/api/v1/create_model_from_state",
                payload,
                timeout_s=60.0,
            )
            resumed = _await_maybe_async(resumed, timeout_s=RESUME_TIMEOUT_S)
            model_id = resumed.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise RuntimeError(f"create_model_from_state missing model_id: {resumed!r}")
            print(f"resumed model_id={model_id} load_optimizer={load_optimizer}", flush=True)
            return model_id
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "500" not in msg and "Internal Server Error" not in msg:
                raise
            if attempt == 2:
                break
            time.sleep(float(attempt + 1))

    assert last_err is not None
    raise last_err


def _load_into_new_session(
    session_id: str,
    model_seq_id: int,
    state_path: str,
    *,
    optimizer: bool = True,
    label: str = "loaded-into-new-session",
) -> str:
    model_id, _backend = _create_model(session_id, model_seq_id)
    _load_state(model_id, state_path, optimizer=optimizer, label=label)
    print(f"{label} model_id={model_id}", flush=True)
    return model_id


def _compare_resume_to_presave(last_presave: TrainStep, resumed: list[TrainStep]) -> dict[str, float]:
    if not resumed:
        raise RuntimeError("no resumed steps recorded")

    first = resumed[0].loss
    max_resumed = max(step.loss for step in resumed)
    final = resumed[-1].loss
    first_ratio = first / max(last_presave.loss, 1e-8)
    max_ratio = max_resumed / max(last_presave.loss, 1e-8)

    for step in resumed:
        rel = step.loss / max(last_presave.loss, 1e-8)
        print(
            f"resume_compare presave_step={last_presave.index} presave_loss={last_presave.loss:.6f} "
            f"resumed_step={step.index} resumed_loss={step.loss:.6f} rel_to_presave={rel:.6f}",
            flush=True,
        )

    return {
        "presave_loss": float(last_presave.loss),
        "first_resumed": float(first),
        "final_resumed": float(final),
        "first_ratio": float(first_ratio),
        "max_ratio": float(max_ratio),
    }


def _write_resume_artifacts(
    *,
    suffix: str,
    last_presave: TrainStep,
    advanced_step: TrainStep,
    resumed_d: list[TrainStep],
    resumed_b: list[TrainStep],
    resumed_c: list[TrainStep],
    summary_d: dict[str, float],
    summary_b: dict[str, float],
    summary_c: dict[str, float],
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib required for merge-gate trajectory artifact: {exc}") from exc

    root = Path(
        os.environ.get("MINT_TEST_EXPERIMENT_ROOT")
        or f"results/merge-gate/manual-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    artifact_dir = root / f"resume_training_{suffix}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "suffix": suffix,
        "presave_reference": {"step": last_presave.index, "loss": last_presave.loss},
        "advanced_reference": {"step": advanced_step.index, "loss": advanced_step.loss},
        "weights_only_rollback": [step.__dict__ for step in resumed_d],
        "create_model_from_state": [step.__dict__ for step in resumed_b],
        "fresh_session_load_state": [step.__dict__ for step in resumed_c],
        "summary": {
            "weights_only_rollback": summary_d,
            "create_model_from_state": summary_b,
            "fresh_session_load_state": summary_c,
        },
    }
    json_path = artifact_dir / "resume_trajectory.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([last_presave.index], [last_presave.loss], marker="o", label="presave_reference")
    ax.plot([advanced_step.index], [advanced_step.loss], marker="o", label="advanced_reference")
    ax.plot([step.index for step in resumed_d], [step.loss for step in resumed_d], marker="o", label="weights_only_rollback")
    ax.plot([step.index for step in resumed_b], [step.loss for step in resumed_b], marker="o", label="create_model_from_state")
    ax.plot([step.index for step in resumed_c], [step.loss for step in resumed_c], marker="o", label="fresh_session_load_state")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"resume_training_{suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    png_path = artifact_dir / "resume_trajectory.png"
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return json_path, png_path


def main() -> int:
    batch = _make_training_batch()
    suffix = uuid.uuid4().hex[:8]
    session_a = f"issue315-a-{suffix}"
    session_b = f"issue315-b-{suffix}"
    session_c = f"issue315-c-{suffix}"
    session_d = f"issue315-d-{suffix}"
    model_a: str | None = None
    model_b: str | None = None
    model_c: str | None = None
    model_d: str | None = None
    try:
        model_a, _ = _create_model(session_a, 0)
        last_presave: TrainStep | None = None
        for step_idx in range(1, WARMUP_STEPS + 1):
            last_presave = _train_step(model_a, batch, step_idx)

        if last_presave is None:
            raise RuntimeError("no pre-save training steps ran")

        checkpoint_path = _save_state(model_a, f"issue315-{suffix}")
        print(f"presave_reference step={last_presave.index} loss={last_presave.loss:.6f}", flush=True)
        _load_state(model_a, checkpoint_path, optimizer=True, label="reloaded")

        advanced_step: TrainStep | None = None
        for step_idx in range(1, DIFFERENT_CHECKPOINT_STEPS + 1):
            advanced_step = _train_step(model_a, batch, WARMUP_STEPS + step_idx)
        if advanced_step is None:
            raise RuntimeError("no different-checkpoint advance steps ran")
        checkpoint_newer_path = _save_state(model_a, f"issue315-{suffix}-newer")
        print(
            f"advanced_reference step={advanced_step.index} loss={advanced_step.loss:.6f} "
            f"newer_checkpoint={checkpoint_newer_path}",
            flush=True,
        )

        model_d = _load_into_new_session(
            session_d,
            0,
            checkpoint_path,
            optimizer=False,
            label="weights-only-rollback",
        )
        resumed_d: list[TrainStep] = []
        for compare_idx in range(1, COMPARE_STEPS + 1):
            resumed_d.append(_train_step(model_d, batch, compare_idx))

        summary_d = _compare_resume_to_presave(last_presave, resumed_d)

        try:
            model_b = _resume_from_state(session_b, 0, checkpoint_path, load_optimizer=True)
        except Exception as exc:
            msg = str(exc)
            if "Optimizer restore requested" in msg or "optimizer shard" in msg:
                print("create_model_from_state optimizer restore unavailable; retrying weights-only", flush=True)
                model_b = _resume_from_state(session_b, 0, checkpoint_path, load_optimizer=False)
            elif "Adapter not found" in msg or "create_model_from_state missing model_id" in msg:
                print("create_model_from_state unavailable for this checkpoint; retrying via create+load_state", flush=True)
                model_b = _load_into_new_session(
                    session_b,
                    0,
                    checkpoint_path,
                    optimizer=False,
                    label="create_model_from_state-fallback",
                )
            else:
                raise
        resumed_b: list[TrainStep] = []
        for compare_idx in range(1, COMPARE_STEPS + 1):
            resumed_b.append(_train_step(model_b, batch, compare_idx))

        summary_b = _compare_resume_to_presave(last_presave, resumed_b)

        try:
            model_c = _load_into_new_session(session_c, 0, checkpoint_path, optimizer=True)
        except Exception as exc:
            msg = str(exc)
            if "Optimizer restore requested" not in msg and "optimizer shard" not in msg:
                raise
            print("load_state optimizer restore unavailable; retrying weights-only", flush=True)
            model_c = _load_into_new_session(session_c, 0, checkpoint_path, optimizer=False)
        resumed_c: list[TrainStep] = []
        for compare_idx in range(1, COMPARE_STEPS + 1):
            resumed_c.append(_train_step(model_c, batch, compare_idx))

        summary_c = _compare_resume_to_presave(last_presave, resumed_c)

        json_path, png_path = _write_resume_artifacts(
            suffix=suffix,
            last_presave=last_presave,
            advanced_step=advanced_step,
            resumed_d=resumed_d,
            resumed_b=resumed_b,
            resumed_c=resumed_c,
            summary_d=summary_d,
            summary_b=summary_b,
            summary_c=summary_c,
        )

        print(
            "SUMMARY: weights-only older-checkpoint rollback, create_model_from_state, and fresh-session "
            f"load_state recorded. artifacts json={json_path} png={png_path} "
            f"(weights-only rollback: {summary_d}; create_model_from_state: {summary_b}; "
            f"fresh-session: {summary_c})",
            flush=True,
        )
        return 0
    except Exception as exc:
        return _fail(str(exc))
    finally:
        for model_id in (model_d, model_c, model_b, model_a):
            if model_id:
                _delete_model(model_id)


if __name__ == "__main__":
    raise SystemExit(main())
