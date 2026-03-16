from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


BASE_URL = (os.environ.get("TINKER_BASE_URL") or "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
BASE_MODEL = os.environ.get("TINKER_BASE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
MODEL_PATH = os.environ.get(
    "TINKER_MODEL_PATH",
    "tinker://ebccebe2-2ccf-584c-822b-11a8ec92c602:train:0/sampler_weights/hotload-base-vllmfix3",
)
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "64"))
LONG_PROMPT_LEN = int(os.environ.get("TINKER_LONG_PROMPT_LEN", "24000"))
LONG_MAX_TOKENS = int(os.environ.get("TINKER_LONG_MAX_TOKENS", "1"))
CHURN_COUNT = int(os.environ.get("TINKER_CHURN_COUNT", "8"))
CHURN_PROMPT_LEN = int(os.environ.get("TINKER_CHURN_PROMPT_LEN", "128"))
CHURN_MAX_TOKENS = int(os.environ.get("TINKER_CHURN_MAX_TOKENS", "8"))
WAIT_BEFORE_CHURN_S = float(os.environ.get("TINKER_WAIT_BEFORE_CHURN_S", "5"))
POLL_SLEEP_S = float(os.environ.get("TINKER_POLL_SLEEP_S", "1.0"))
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "2400"))
MONITOR_INTERVAL_S = float(os.environ.get("TINKER_MONITOR_INTERVAL_S", "0.5"))
MONITOR_NODE_IPS = [
    x.strip() for x in os.environ.get("TINKER_MONITOR_NODE_IPS", "192.168.37.187,192.168.37.188").split(",") if x.strip()
]
RAY_NAMESPACE = os.environ.get("TINKER_RAY_NAMESPACE", "tinker_issue328")
OUTPUT_JSON = os.environ.get("TINKER_OUTPUT_JSON", f"/tmp/issue328_repro_{int(time.time())}.json")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float = 120.0) -> tuple[int, dict[str, Any]]:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:2000]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": repr(data)}
    return resp.status_code, data


def _retrieve_future_once(request_id: str) -> tuple[int, dict[str, Any]]:
    return _post_json("/api/v1/retrieve_future", {"request_id": request_id}, timeout_s=60.0)


def _create_session(tag: str) -> str:
    status, data = _post_json(
        "/api/v1/create_session",
        {"tags": [tag], "user_metadata": {}, "sdk_version": "reproduce_issue_328"},
        timeout_s=60.0,
    )
    if status != 200:
        raise RuntimeError(f"create_session returned {status}: {data!r}")
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {data!r}")
    return session_id


def _create_sampling_session(session_id: str) -> str:
    status, data = _post_json(
        "/api/v1/create_sampling_session",
        {
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": BASE_MODEL,
            "model_path": MODEL_PATH,
            "lora_rank": LORA_RANK,
        },
        timeout_s=1800.0,
    )
    if status != 200:
        raise RuntimeError(f"create_sampling_session returned {status}: {data!r}")
    sampling_session_id = data.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {data!r}")
    return sampling_session_id


def _submit_asample(
    *,
    sampling_session_id: str,
    seq_id: int,
    prompt_tokens: list[int],
    max_tokens: int,
    prompt_logprobs: bool,
) -> str:
    status, data = _post_json(
        "/api/v1/asample",
        {
            "sampling_session_id": sampling_session_id,
            "seq_id": seq_id,
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
            "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "top_p": 1.0},
            "prompt_logprobs": bool(prompt_logprobs),
        },
        timeout_s=120.0,
    )
    if status != 200:
        raise RuntimeError(f"asample returned {status}: {data!r}")
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {data!r}")
    return request_id


@dataclass
class FutureOutcome:
    request_id: str
    terminal_status: int
    body: dict[str, Any]
    elapsed_s: float


def _poll_future(request_id: str, *, timeout_s: float) -> FutureOutcome:
    deadline = time.time() + timeout_s
    start = time.time()
    last_status = 0
    last_body: dict[str, Any] = {}
    while time.time() < deadline:
        status, body = _retrieve_future_once(request_id)
        last_status = status
        last_body = body
        if status != 408:
            return FutureOutcome(request_id=request_id, terminal_status=status, body=body, elapsed_s=time.time() - start)
        time.sleep(POLL_SLEEP_S)
    raise TimeoutError(
        f"retrieve_future timed out after {timeout_s:.1f}s: request_id={request_id} last_status={last_status} last_body={last_body!r}"
    )


def _make_tokens(length: int, *, seed: int) -> list[int]:
    # Use a deterministic varying pattern to reduce prefix-cache triviality while staying valid token ids.
    return [((seed + i) % 1000) + 100 for i in range(length)]


def _init_ray() -> Any:
    import ray

    return ray.init(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True, logging_level="ERROR")


def _start_monitors() -> tuple[list[Any], Any]:
    import ray

    @ray.remote(num_cpus=0.1)
    class GpuMonitor:
        def __init__(self, interval_s: float) -> None:
            self.interval_s = interval_s
            self.samples: list[dict[str, Any]] = []
            self.peaks: dict[int, int] = {}
            self.running = False
            self.thread: threading.Thread | None = None

        def start(self) -> None:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

        def _loop(self) -> None:
            import subprocess

            while self.running:
                ts = time.time()
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                    )
                    rows = []
                    for line in out.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        idx_s, mem_s = [part.strip() for part in line.split(",", 1)]
                        idx = int(idx_s)
                        mem = int(mem_s)
                        rows.append({"gpu": idx, "mem_mib": mem})
                        prev = self.peaks.get(idx, 0)
                        if mem > prev:
                            self.peaks[idx] = mem
                    self.samples.append({"ts": ts, "gpus": rows})
                except Exception as e:
                    self.samples.append({"ts": ts, "error": f"{type(e).__name__}: {e}"})
                time.sleep(self.interval_s)

        def stop(self) -> dict[str, Any]:
            self.running = False
            if self.thread is not None:
                self.thread.join(timeout=5)
            return {"samples": self.samples, "peak_by_gpu_mib": self.peaks}

    monitors = []
    for node_ip in MONITOR_NODE_IPS:
        actor = GpuMonitor.options(resources={f"node:{node_ip}": 0.001}).remote(MONITOR_INTERVAL_S)
        ray.get(actor.start.remote())
        monitors.append((node_ip, actor))
    return monitors, ray


def _run_churn(idx: int) -> dict[str, Any]:
    session_id = _create_session(f"issue328-churn-{idx}")
    sampling_session_id = _create_sampling_session(session_id)
    request_id = _submit_asample(
        sampling_session_id=sampling_session_id,
        seq_id=0,
        prompt_tokens=_make_tokens(CHURN_PROMPT_LEN, seed=1000 + idx * 97),
        max_tokens=CHURN_MAX_TOKENS,
        prompt_logprobs=False,
    )
    outcome = _poll_future(request_id, timeout_s=POLL_TIMEOUT_S)
    return {
        "idx": idx,
        "session_id": session_id,
        "sampling_session_id": sampling_session_id,
        "request_id": request_id,
        "outcome": asdict(outcome),
    }


def main() -> int:
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "base_url": BASE_URL,
        "base_model": BASE_MODEL,
        "model_path": MODEL_PATH,
        "lora_rank": LORA_RANK,
        "ray_namespace": RAY_NAMESPACE,
        "monitor_node_ips": MONITOR_NODE_IPS,
        "started_at": time.time(),
    }

    _init_ray()
    monitors, ray = _start_monitors()
    try:
        long_session_id = _create_session("issue328-long")
        long_sampling_session_id = _create_sampling_session(long_session_id)
        long_request_id = _submit_asample(
            sampling_session_id=long_sampling_session_id,
            seq_id=0,
            prompt_tokens=_make_tokens(LONG_PROMPT_LEN, seed=17),
            max_tokens=LONG_MAX_TOKENS,
            prompt_logprobs=True,
        )
        result["long_request"] = {
            "session_id": long_session_id,
            "sampling_session_id": long_sampling_session_id,
            "request_id": long_request_id,
        }
        time.sleep(WAIT_BEFORE_CHURN_S)
        initial_status, initial_body = _retrieve_future_once(long_request_id)
        result["long_request"]["status_after_wait"] = {"status": initial_status, "body": initial_body}

        churn_results: list[dict[str, Any]] = []
        churn_errors: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=CHURN_COUNT) as pool:
            futures = [pool.submit(_run_churn, idx) for idx in range(CHURN_COUNT)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    churn_results.append(future.result())
                except Exception as e:
                    churn_errors.append({"error": f"{type(e).__name__}: {e}"})
        result["churn_results"] = churn_results
        result["churn_errors"] = churn_errors

        try:
            long_outcome = _poll_future(long_request_id, timeout_s=POLL_TIMEOUT_S)
            result["long_request"]["terminal"] = asdict(long_outcome)
        except Exception as e:
            result["long_request"]["terminal_error"] = f"{type(e).__name__}: {e}"

        try:
            result["named_actors_after"] = ray.util.list_named_actors(all_namespaces=True)
        except Exception as e:
            result["named_actors_after_error"] = f"{type(e).__name__}: {e}"
    finally:
        monitor_results = {}
        for node_ip, actor in monitors:
            try:
                monitor_results[node_ip] = ray.get(actor.stop.remote())
            except Exception as e:
                monitor_results[node_ip] = {"error": f"{type(e).__name__}: {e}"}
        result["monitor_results"] = monitor_results
        result["finished_at"] = time.time()
        Path(OUTPUT_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"output_json": OUTPUT_JSON}, ensure_ascii=True), flush=True)

    def _body_text(body: Any) -> str:
        if isinstance(body, dict):
            return json.dumps(body, ensure_ascii=True)
        return str(body)

    def _is_issue_328_signal(text: str) -> bool:
        return any(
            needle in text
            for needle in (
                "multinode_vllm_ray_get_failed",
                "ActorDiedError",
                "EngineDeadError",
                "RayActorError",
                "killed by `ray.kill`",
                "killed by `ray.kill'.",
            )
        )

    def _peak_memory_by_node() -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for node_ip, payload in (result.get("monitor_results") or {}).items():
            peaks = payload.get("peak_by_gpu_mib") if isinstance(payload, dict) else None
            if not isinstance(peaks, dict):
                continue
            normalized = {str(gpu): int(mem) for gpu, mem in peaks.items()}
            if normalized:
                out[str(node_ip)] = normalized
        return out

    generic_failure = False
    issue_328_signal = False
    for rec in result.get("churn_results", []):
        outcome = rec.get("outcome") or {}
        if int(outcome.get("terminal_status", 0)) != 200:
            generic_failure = True
        text = _body_text(outcome.get("body"))
        if "error" in text:
            generic_failure = True
        if _is_issue_328_signal(text):
            issue_328_signal = True
    long_terminal = (result.get("long_request") or {}).get("terminal") or {}
    if int(long_terminal.get("terminal_status", 0)) != 200:
        generic_failure = True
    long_text = _body_text(long_terminal.get("body"))
    if "error" in long_text:
        generic_failure = True
    if _is_issue_328_signal(long_text):
        issue_328_signal = True
    if result.get("churn_errors"):
        generic_failure = True
        if any(_is_issue_328_signal(str(x)) for x in result["churn_errors"]):
            issue_328_signal = True

    result["summary"] = {
        "generic_failure": generic_failure,
        "issue_328_signal": issue_328_signal,
        "peak_memory_by_node_mib": _peak_memory_by_node(),
    }
    Path(OUTPUT_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": OUTPUT_JSON, "issue_328_signal": issue_328_signal}, ensure_ascii=True), flush=True)
    return 0 if issue_328_signal else (1 if generic_failure else 2)


if __name__ == "__main__":
    raise SystemExit(main())
