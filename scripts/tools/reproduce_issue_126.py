import math
import os
import sys
import time
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

BASE_MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-0.6B")
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "8"))

CREATE_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_MODEL_TIMEOUT_S", "3600"))
FWDBWD_TIMEOUT_S = float(os.environ.get("TINKER_FORWARD_BACKWARD_TIMEOUT_S", "3600"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-dict json: {type(data)}")
    return data


def _poll_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
            return data
        if resp.status_code == 408:
            time.sleep(2)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=60)
    except Exception:
        pass


def _get_loss_and_logprobs(fwd_bwd_result: dict[str, Any]) -> tuple[float, list[float]]:
    outs = fwd_bwd_result.get("loss_fn_outputs")
    if not isinstance(outs, list) or not outs or not isinstance(outs[0], dict):
        raise RuntimeError(f"unexpected forward_backward payload (missing loss_fn_outputs): {fwd_bwd_result!r}")
    loss_obj = outs[0].get("loss")
    logprobs_obj = outs[0].get("logprobs")
    if not isinstance(loss_obj, dict) or not isinstance(logprobs_obj, dict):
        raise RuntimeError(f"unexpected forward_backward payload (missing loss/logprobs): {fwd_bwd_result!r}")
    loss_data = loss_obj.get("data")
    logprobs_data = logprobs_obj.get("data")
    if not (isinstance(loss_data, list) and len(loss_data) == 1 and isinstance(loss_data[0], (int, float))):
        raise RuntimeError(f"unexpected loss.data: {loss_data!r}")
    if not (isinstance(logprobs_data, list) and all(isinstance(x, (int, float)) for x in logprobs_data)):
        raise RuntimeError(f"unexpected logprobs.data: {logprobs_data!r}")
    return float(loss_data[0]), [float(x) for x in logprobs_data]


def main() -> int:
    model_id: str | None = None
    try:
        session_id = f"repro-126-{uuid.uuid4().hex[:8]}"
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
            created = _poll_future(str(created["request_id"]), timeout_s=CREATE_TIMEOUT_S)
        if "error" in created:
            return _fail(f"create_model failed: {created.get('error')!r}")
        model_id = created.get("model_id")
        if not model_id:
            return _fail(f"create_model missing model_id: {created!r}")
        if created.get("backend") != "peft":
            return _fail(f"expected dense backend=peft. got {created.get('backend')!r} (model={BASE_MODEL!r})")

        # Build one Datum with positive weights. We check that cross_entropy returns the SUM of
        # weighted token losses, not the average over weights.sum().
        tokens = [10, 11, 12, 13, 14, 15]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [1.0] * len(target_tokens)

        datum = {
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
            },
        }

        fb = _post_json(
            f"{BASE_URL}/api/v1/forward_backward",
            {
                "model_id": model_id,
                "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy", "loss_fn_config": {}},
            },
            timeout_s=60.0,
        )
        if "request_id" in fb:
            fb = _poll_future(str(fb["request_id"]), timeout_s=FWDBWD_TIMEOUT_S)
        if "error" in fb:
            return _fail(f"forward_backward failed: {fb.get('error')!r}")

        loss, logprobs = _get_loss_and_logprobs(fb)
        if len(logprobs) != len(weights):
            return _fail(f"logprobs length {len(logprobs)} != weights length {len(weights)}")

        expected_sum = -sum(lp * w for lp, w in zip(logprobs, weights))
        if not (math.isfinite(loss) and math.isfinite(expected_sum)):
            return _fail(f"non-finite loss values: loss={loss}, expected_sum={expected_sum}")

        # Large factor mismatch indicates unintended averaging by weights.sum().
        if abs(loss - expected_sum) > 1e-3:
            expected_mean = expected_sum / sum(weights)
            return _fail(
                "cross_entropy loss mismatch: expected sum(-logp * weights)\n"
                f"  loss={loss:.6f}\n"
                f"  expected_sum={expected_sum:.6f}\n"
                f"  expected_mean={expected_mean:.6f} (if averaged)\n"
            )

        # importance_sampling should use old_logprobs and advantages, not reduce to cross_entropy.
        old_logprobs = [0.0] * len(target_tokens)
        advantages = [0.0, 1.0, -2.0, 0.5, 0.0]
        if len(advantages) != len(target_tokens):
            return _fail("internal repro error: advantages length mismatch")

        datum_rl = {
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
                "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
                "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
            },
        }

        fb_rl = _post_json(
            f"{BASE_URL}/api/v1/forward_backward",
            {
                "model_id": model_id,
                "forward_backward_input": {"data": [datum_rl], "loss_fn": "importance_sampling", "loss_fn_config": {}},
            },
            timeout_s=60.0,
        )
        if "request_id" in fb_rl:
            fb_rl = _poll_future(str(fb_rl["request_id"]), timeout_s=FWDBWD_TIMEOUT_S)
        if "error" in fb_rl:
            return _fail(f"forward_backward(importance_sampling) failed: {fb_rl.get('error')!r}")

        loss_rl, new_logprobs = _get_loss_and_logprobs(fb_rl)
        if len(new_logprobs) != len(target_tokens):
            return _fail(f"importance_sampling logprobs length mismatch: {len(new_logprobs)}")

        expected_rl = 0.0
        for lp_new, lp_old, adv, wt in zip(new_logprobs, old_logprobs, advantages, weights):
            log_ratio = lp_new - lp_old
            log_ratio = max(-20.0, min(20.0, log_ratio))
            ratio = math.exp(log_ratio)
            expected_rl += -(ratio * adv * wt)

        if not (math.isfinite(loss_rl) and math.isfinite(expected_rl)):
            return _fail(f"non-finite RL loss values: loss={loss_rl}, expected={expected_rl}")

        if abs(loss_rl - expected_rl) > 1e-2:
            return _fail(
                "importance_sampling loss mismatch\n"
                f"  loss={loss_rl:.6f}\n"
                f"  expected={expected_rl:.6f}\n"
            )

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id:
            _delete_model(model_id)


if __name__ == "__main__":
    raise SystemExit(main())
