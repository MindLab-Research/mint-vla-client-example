import os
import shlex
import subprocess
import time
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
SSH_HOST = os.environ.get("MINT_SSH_HOST", "mint-dev")

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
BASE_MODEL = os.environ.get("MINT_MODEL", DEFAULT_MODEL)
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "8"))

LR1 = float(os.environ.get("MINT_LR1", "1e-4"))
LR2 = float(os.environ.get("MINT_LR2", "5e-5"))

CHECKPOINTS_ROOT = os.environ.get(
    "MINT_CHECKPOINTS_ROOT", "/vePFS-Mindverse/share/mint_checkpoints"
)


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text[:400]!r}")
    return r.json()


def _poll_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 408:
            time.sleep(1.0)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _ssh(cmd: str) -> str:
    out = subprocess.check_output(["ssh", SSH_HOST, cmd], text=True).strip()
    return out


def _model_info(model_id: str) -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"GET /models/{model_id} -> {r.status_code}: {r.text[:400]!r}")
    return r.json()


def _train_step(model_id: str, *, lr: float) -> dict[str, Any]:
    tokens = [1, 2, 3, 4]
    targets = [2, 3, 4]
    weights = [1.0, 1.0, 1.0]
    datum = {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": targets},
            "weights": {"data": weights},
        },
    }
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"},
        "adam_params": {"learning_rate": lr},
    }
    out = _post_json(f"{BASE_URL}/api/v1/train_step", payload, timeout_s=60.0)
    if "request_id" in out:
        out = _poll_future(str(out["request_id"]), timeout_s=600.0)
    return out


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", flush=True)
    return 1


def main() -> int:
    model_id: str | None = None
    try:
        session_id = f"repro-187-{uuid.uuid4().hex[:8]}"
        created = _post_json(
            f"{BASE_URL}/api/v1/create_model",
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": BASE_MODEL,
                "lora_config": {"rank": LORA_RANK},
            },
            timeout_s=60.0,
        )
        if "request_id" in created:
            created = _poll_future(str(created["request_id"]), timeout_s=1800.0)

        model_id = created.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return _fail(f"create_model missing model_id: {created!r}")

        # Step to ensure optimizer/step metadata is non-trivial.
        _train_step(model_id, lr=LR1)
        _train_step(model_id, lr=LR1)
        info_before = _model_info(model_id)

        saved = _post_json(
            f"{BASE_URL}/api/v1/save_state",
            {"model_id": model_id, "path": "step-2"},
            timeout_s=60.0,
        )
        if "request_id" in saved:
            saved = _poll_future(str(saved["request_id"]), timeout_s=1800.0)
        training_uri = saved.get("path")
        training_fs_path = saved.get("filesystem_path")
        if not isinstance(training_uri, str) or not training_uri:
            return _fail(f"save_state missing path: {saved!r}")
        if not isinstance(training_fs_path, str) or not training_fs_path:
            return _fail(f"save_state missing filesystem_path: {saved!r}")

        opt_path = os.path.join(training_fs_path, "optimizer.pt")
        out = _ssh(f"test -f {shlex.quote(opt_path)} && echo ok || echo missing")
        if out.strip() != "ok":
            return _fail(f"save_state missing optimizer.pt: {opt_path}")

        sampler = _post_json(
            f"{BASE_URL}/api/v1/save_weights_for_sampler",
            {"model_id": model_id, "path": "sample-2"},
            timeout_s=60.0,
        )
        if "request_id" in sampler:
            sampler = _poll_future(str(sampler["request_id"]), timeout_s=1800.0)

        sampler_uri = sampler.get("path")
        if not isinstance(sampler_uri, str) or not sampler_uri:
            return _fail(f"save_weights_for_sampler missing path: {sampler!r}")

        sampler_fs_path = os.path.join(CHECKPOINTS_ROOT, "anonymous", model_id, "sample-2")
        sampler_meta = os.path.join(sampler_fs_path, "metadata.json")
        out = _ssh(f"test -f {shlex.quote(sampler_meta)} && echo ok || echo missing")
        if out.strip() != "ok":
            return _fail(f"save_weights_for_sampler missing metadata.json: {sampler_meta}")
        out = _ssh(
            f"test -f {shlex.quote(os.path.join(sampler_fs_path, 'optimizer.pt'))} && echo bad || echo ok"
        )
        if out.strip() != "ok":
            return _fail(f"sampler checkpoint unexpectedly contains optimizer.pt: {sampler_fs_path}")

        # Negative path: sampler checkpoint must be rejected for optimizer restore.
        bad = requests.post(
            f"{BASE_URL}/api/v1/load_state",
            headers=_headers(),
            json={"model_id": model_id, "path": sampler_uri, "optimizer": True},
            timeout=30,
        )
        if bad.status_code != 400:
            return _fail(f"expected 400 for load_state_with_optimizer(sampler); got {bad.status_code}: {bad.text!r}")

        # Mutate state so we can verify restore of step+lr.
        _train_step(model_id, lr=LR2)
        info_mutated = _model_info(model_id)

        good = _post_json(
            f"{BASE_URL}/api/v1/load_state",
            {"model_id": model_id, "path": training_uri, "optimizer": True},
            timeout_s=60.0,
        )
        if "request_id" in good:
            good = _poll_future(str(good["request_id"]), timeout_s=1800.0)

        info_after = _model_info(model_id)
        if int(info_after.get("current_step") or 0) != int(info_before.get("current_step") or 0):
            return _fail(
                f"current_step not restored: before={info_before.get('current_step')!r} "
                f"mutated={info_mutated.get('current_step')!r} after={info_after.get('current_step')!r}"
            )
        if abs(float(info_after.get("learning_rate") or 0.0) - LR1) > 1e-12:
            print(f"DEBUG info_before={info_before!r}", flush=True)
            print(f"DEBUG info_mutated={info_mutated!r}", flush=True)
            print(f"DEBUG info_after={info_after!r}", flush=True)
            return _fail(
                f"learning_rate not restored: before={info_before.get('learning_rate')!r} "
                f"mutated={info_mutated.get('learning_rate')!r} after={info_after.get('learning_rate')!r}"
            )

        print("PASS", flush=True)
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id:
            try:
                requests.delete(
                    f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=60
                )
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
