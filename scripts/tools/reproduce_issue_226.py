import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# The fix for this issue is about actor accounting and reconciliation, which is
# fundamentally Ray-backed. The repro creates named Ray actors in the server's
# namespace, then exercises server endpoints over HTTP.
RAY_NAMESPACE = (
    os.environ.get("TINKER_RAY_NAMESPACE")
    or os.environ.get("MINT_RAY_NAMESPACE")
    or ""
).strip()

SSH_HOST = os.environ.get("TINKER_DEV_SSH_HOST", "mint-dev").strip() or "mint-dev"

ISSUE_NUMBER = 226
REMOTE_SERVER_ROOT = os.environ.get(
    "TINKER_ISSUE_REMOTE_ROOT", f"/root/tinker_project/tinker-server-issue-{ISSUE_NUMBER}"
).rstrip("/")

POLL_DELAY_S = float(os.environ.get("TINKER_POLL_DELAY_S", "0.5"))


def _headers() -> dict[str, str]:
    # Dev servers typically disable auth, but keep header support for parity.
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _get_json(url: str, *, timeout_s: float) -> Any:
    r = requests.get(url, headers=_headers(), timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _post(url: str, payload: dict, *, timeout_s: float) -> requests.Response:
    return requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _ssh_run(argv: list[str], *, input_text: str | None = None, timeout_s: float = 60.0) -> str:
    cmd = ["ssh", SSH_HOST, *argv]
    p = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"ssh_failed rc={p.returncode} cmd={cmd!r}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p.stdout


def _ssh_python(code: str, *, timeout_s: float = 60.0) -> str:
    _require(RAY_NAMESPACE, "TINKER_RAY_NAMESPACE (or MINT_RAY_NAMESPACE) must be set for this repro")
    return _ssh_run(
        [
            "env",
            f"TINKER_RAY_NAMESPACE={RAY_NAMESPACE}",
            f"MINT_RAY_NAMESPACE={RAY_NAMESPACE}",
            "python3",
            "-",
        ],
        input_text=code,
        timeout_s=timeout_s,
    )


def _wait_healthz_ready(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=5.0)
            if r.status_code == 200 and r.json().get("status") == "ready":
                return
            last_err = f"http_status={r.status_code} body={r.text[:200]!r}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(POLL_DELAY_S)
    raise RuntimeError(f"healthz_not_ready timeout_s={timeout_s} last_err={last_err}")


@dataclass(frozen=True)
class _NamedActor:
    name: str
    kind: str


def _ssh_create_named_actor(actor: _NamedActor, *, block_s: int | None = None) -> None:
    # Create a detached named actor in the server namespace, without registering it in ModelActorRegistry.
    # Optionally start a long-running task to force __ray_ready__ probes to time out.
    block_s_expr = "None" if block_s is None else str(int(block_s))
    code = f"""
import os
import time
import ray

ns = os.environ["TINKER_RAY_NAMESPACE"]
addr = os.environ.get("RAY_ADDRESS", "").strip()
if not addr:
    raise RuntimeError("RAY_ADDRESS is required")
ray.init(address=addr, namespace=ns, ignore_reinit_error=True)

@ray.remote
class ReproActor:
    def block(self, seconds: int) -> str:
        time.sleep(int(seconds))
        return "done"

name = {json.dumps(actor.name)}
try:
    existing = ray.get_actor(name, namespace=ns)
except ValueError:
    existing = None
if existing is not None:
    ray.kill(existing, no_restart=True)
    # Ensure name is released.
    for _ in range(10):
        time.sleep(0.2)
        try:
            ray.get_actor(name, namespace=ns)
        except ValueError:
            break

a = ReproActor.options(name=name, lifetime="detached").remote()
block_s = {block_s_expr}
if block_s is not None:
    # Fire-and-forget long actor task so __ray_ready__ queues behind it.
    a.block.remote(int(block_s))

kind = {json.dumps(actor.kind)}
print(f"created name={{name}} namespace={{ns}} kind={{kind}} block_s={{block_s}}")
"""
    _ssh_python(code, timeout_s=60.0)


def _ssh_kill_named_actor(name: str) -> None:
    code = f"""
import os
import ray

ns = os.environ["TINKER_RAY_NAMESPACE"]
addr = os.environ.get("RAY_ADDRESS", "").strip()
if not addr:
    raise RuntimeError("RAY_ADDRESS is required")
ray.init(address=addr, namespace=ns, ignore_reinit_error=True)

name = {json.dumps(name)}
try:
    a = ray.get_actor(name, namespace=ns)
except ValueError:
    print(f"missing name={{name}} namespace={{ns}}")
else:
    ray.kill(a, no_restart=True)
    print(f"killed name={{name}} namespace={{ns}}")
"""
    _ssh_python(code, timeout_s=60.0)


def _actors_by_name(actor_name: str) -> list[dict[str, Any]]:
    payload = _get_json(f"{BASE_URL}/api/v1/actors?type=vllm", timeout_s=30.0)
    actors = payload.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError(f"invalid /actors payload keys={sorted(payload.keys())}")
    return [a for a in actors if isinstance(a, dict) and a.get("actor_name") == actor_name]


def main() -> int:
    errors: list[str] = []

    print(f"base_url={BASE_URL}")
    print(f"ray_namespace={RAY_NAMESPACE!r}")
    print(f"ssh_host={SSH_HOST} remote_root={REMOTE_SERVER_ROOT}")

    # Ensure server is reachable before starting.
    try:
        _wait_healthz_ready(timeout_s=30.0)
    except Exception as e:
        return _fail(f"healthz_unreachable: {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # Repro 1: /actors/kill should not fall back to Ray named-actor registry when
    # ModelActorRegistry has no VLLM entries. Before the fix, the endpoint kills the
    # actor anyway (silent fallback). After the fix, it should surface the
    # mismatch as an error (409) rather than silently switching registries.
    # -------------------------------------------------------------------------
    named_only = _NamedActor(name="tinker_vllm_repro_226_named_only", kind="named_only")
    try:
        _ssh_create_named_actor(named_only)
        _require(
            not _actors_by_name(named_only.name),
            f"precondition_failed: {named_only.name} unexpectedly appears in ModelActorRegistry",
        )

        r = _post(
            f"{BASE_URL}/api/v1/actors/kill",
            {"actor_type": "vllm", "model_name": None},
            timeout_s=30.0,
        )
        if r.status_code != 409:
            errors.append(
                f"actors_kill_unexpected_status expected=409 got={r.status_code} body={r.text[:200]!r}"
            )
    except Exception as e:
        errors.append(f"actors_kill_repro_failed: {type(e).__name__}: {e}")
    finally:
        # Cleanup: do not leave detached actors around even when the repro fails.
        try:
            _ssh_kill_named_actor(named_only.name)
        except Exception as e:
            errors.append(f"cleanup_failed actor={named_only.name}: {type(e).__name__}: {e}")

    if errors:
        print("\n".join(f"FAIL: {e}" for e in errors), file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
