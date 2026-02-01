import os
import sys
from pathlib import Path
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL")
if not BASE_URL:
    port = os.environ.get("TINKER_PORT", "8000")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get(path: str, *, timeout_s: float, expect_status: int = 200) -> requests.Response:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    if resp.status_code != expect_status:
        raise RuntimeError(f"GET {path} -> {resp.status_code} (expected {expect_status}): {resp.text[:500]!r}")
    return resp


def _get_json(path: str, *, timeout_s: float, expect_status: int = 200) -> dict[str, Any]:
    resp = _get(path, timeout_s=timeout_s, expect_status=expect_status)
    if expect_status != 200:
        return {}
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {path} returned non-dict json: {type(data)}")
    return data


def _post(path: str, payload: dict[str, Any], *, timeout_s: float, expect_status: int = 200) -> requests.Response:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != expect_status:
        raise RuntimeError(f"POST {path} -> {resp.status_code} (expected {expect_status}): {resp.text[:500]!r}")
    return resp


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float, expect_status: int = 200) -> dict[str, Any]:
    resp = _post(path, payload, timeout_s=timeout_s, expect_status=expect_status)
    if expect_status != 200:
        return {}
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(data)}")
    return data


def _load_expected_config() -> dict[str, Any]:
    p = Path(__file__).resolve().parents[2] / "configs" / "issue_136.toml"
    try:
        import tomllib
    except Exception:  # pragma: no cover
        import tomli as tomllib
    d = tomllib.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise RuntimeError(f"expected dict at {p}")
    return d


def _get_in(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _expect_server_info_config(info: dict[str, Any], expected: dict[str, Any]) -> None:
    cfg = info.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"server_info missing config dict: {cfg!r}")

    exp_max_loras = int(_get_in(expected, "server.max_loras"))
    exp_max_cpu_loras = int(_get_in(expected, "server.max_cpu_loras"))
    exp_max_lora_rank = int(_get_in(expected, "server.max_lora_rank"))

    got_max_loras = cfg.get("max_loras")
    got_max_cpu_loras = cfg.get("max_cpu_loras")
    got_max_lora_rank = cfg.get("max_lora_rank")

    if got_max_loras != exp_max_loras:
        raise RuntimeError(f"server_info config.max_loras={got_max_loras!r} expected {exp_max_loras}")
    if got_max_cpu_loras != exp_max_cpu_loras:
        raise RuntimeError(f"server_info config.max_cpu_loras={got_max_cpu_loras!r} expected {exp_max_cpu_loras}")
    if got_max_lora_rank != exp_max_lora_rank:
        raise RuntimeError(f"server_info config.max_lora_rank={got_max_lora_rank!r} expected {exp_max_lora_rank}")

    exp_inflight = int(_get_in(expected, "sampling.max_inflight_sample_tasks"))
    exp_concurrent = int(_get_in(expected, "sampling.max_concurrent_samples_per_request"))
    got_inflight = cfg.get("sampling_max_inflight_sample_tasks")
    got_concurrent = cfg.get("sampling_max_concurrent_samples_per_request")
    if got_inflight != exp_inflight:
        raise RuntimeError(
            f"server_info config.sampling_max_inflight_sample_tasks={got_inflight!r} expected {exp_inflight}"
        )
    if got_concurrent != exp_concurrent:
        raise RuntimeError(
            f"server_info config.sampling_max_concurrent_samples_per_request={got_concurrent!r} expected {exp_concurrent}"
        )


def main() -> int:
    try:
        expected = _load_expected_config()

        _get_json("/api/v1/healthz", timeout_s=10.0)
        info = _get_json("/api/v1/server_info", timeout_s=10.0)
        _expect_server_info_config(info, expected)

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
