from __future__ import annotations

import asyncio
import itertools
import structlog
import json
import time
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mint_server.ray.runtime_env import bootstrap_runtime_pythonpath, validate_runtime_env_layout


OPENPI_FAST_WORKER_PROTOCOL_VERSION = 1
OPENPI_FAST_WORKER_MODULE = f"{__package__}.openpi_fast_worker"

logger = structlog.get_logger(__name__)

class OpenPIFastWorkerError(RuntimeError):
    pass


class OpenPIFastWorkerProtocolError(OpenPIFastWorkerError):
    pass


class OpenPIFastWorkerRemoteError(OpenPIFastWorkerError):
    def __init__(self, *, error_type: str, message: str, traceback_text: str | None = None) -> None:
        self.error_type = error_type
        self.traceback_text = traceback_text
        super().__init__(f"{error_type}: {message}")


def _default_pythonpath() -> tuple[str, ...]:
    current = os.environ.get("PYTHONPATH", "").strip()
    return tuple(item for item in current.split(os.pathsep) if item)


def _default_mint_root() -> Path:
    mint_code_root = (os.environ.get("MINT_CODE_ROOT") or "").strip()
    if mint_code_root:
        return Path(mint_code_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _require_existing_executable(path: Path, *, label: str) -> str:
    resolved = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        mode = resolved.stat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"OpenPI FAST {label} does not exist: {resolved}") from exc
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"OpenPI FAST {label} is not executable: {resolved}")
    return str(resolved)


def _merge_pythonpath(entries: tuple[str, ...], current: str | None) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in (*entries, *(current.split(os.pathsep) if current else ())):
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return os.pathsep.join(ordered)


@dataclass(frozen=True)
class OpenPIFastRuntimeSpec:
    python_executable: str = sys.executable
    worker_module: str = OPENPI_FAST_WORKER_MODULE
    pythonpath: tuple[str, ...] = field(default_factory=_default_pythonpath)
    startup_timeout_s: float = 30.0
    request_timeout_s: float = 300.0
    create_session_timeout_s: float = 300.0
    save_weights_timeout_s: float = 300.0
    load_weights_timeout_s: float = 300.0
    cwd: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "OpenPIFastRuntimeSpec":
        mint_root = _default_mint_root()
        pythonpath_env = os.environ.get("MINT_OPENPI_FAST_PYTHONPATH", "").strip()
        pythonpath = tuple(s for s in pythonpath_env.split(os.pathsep) if s)
        if not pythonpath:
            pythonpath = tuple(
                s
                for s in bootstrap_runtime_pythonpath(
                    os.environ,
                    repo_root=str(mint_root),
                ).split(os.pathsep)
                if s
            )
        request_timeout_s = float(os.environ.get("MINT_OPENPI_FAST_REQUEST_TIMEOUT_S", "300"))
        python_executable = (os.environ.get("MINT_OPENPI_FAST_PYTHON") or "").strip()
        if python_executable:
            python_executable = _require_existing_executable(
                Path(python_executable),
                label="runtime python",
            )
        else:
            env_root = (os.environ.get("PFS_RUNTIME_ENV_ROOT") or "").strip()
            if not env_root:
                raise RuntimeError("PFS_RUNTIME_ENV_ROOT is required for OpenPI FAST runtime")
            python_executable = validate_runtime_env_layout(
                env_root,
                require_host_python=True,
            ).host_python
        return cls(
            python_executable=python_executable,
            worker_module=os.environ.get(
                "MINT_OPENPI_FAST_WORKER_MODULE",
                OPENPI_FAST_WORKER_MODULE,
            ),
            pythonpath=pythonpath,
            startup_timeout_s=float(os.environ.get("MINT_OPENPI_FAST_STARTUP_TIMEOUT_S", "30")),
            request_timeout_s=request_timeout_s,
            create_session_timeout_s=float(
                os.environ.get("MINT_OPENPI_FAST_CREATE_SESSION_TIMEOUT_S", str(request_timeout_s))
            ),
            save_weights_timeout_s=float(
                os.environ.get("MINT_OPENPI_FAST_SAVE_TIMEOUT_S", str(request_timeout_s))
            ),
            load_weights_timeout_s=float(
                os.environ.get("MINT_OPENPI_FAST_LOAD_TIMEOUT_S", str(request_timeout_s))
            ),
            cwd=os.environ.get("MINT_OPENPI_FAST_CWD") or str(mint_root),
        )

    def build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Do not inherit the parent server PYTHONPATH; helper venvs can shadow
        # the OpenPI runtime torch/torchvision pair that the child must use.
        env["PYTHONPATH"] = _merge_pythonpath(self.pythonpath, None)
        env.update(self.extra_env)
        # OpenPI protocol workers are subprocesses, not service actors. They
        # already receive a complete env from the parent actor and must not try
        # to hydrate from ConfigActor while importing mint_server.backend.
        env["MINT_CONFIG_ACTOR_HYDRATE"] = "0"
        return env


class OpenPIFastWorkerClient:
    def __init__(
        self,
        *,
        process: asyncio.subprocess.Process,
        spec: OpenPIFastRuntimeSpec,
    ) -> None:
        self._process = process
        self._spec = spec
        self._request_ids = itertools.count(1)
        self._request_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def start(cls, spec: OpenPIFastRuntimeSpec) -> "OpenPIFastWorkerClient":
        process = await asyncio.create_subprocess_exec(
            spec.python_executable,
            "-m",
            spec.worker_module,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            cwd=spec.cwd,
            env=spec.build_env(),
        )
        client = cls(process=process, spec=spec)
        await client._await_ready()
        return client

    async def _read_message(self, timeout_s: float) -> dict[str, Any]:
        stdout = self._process.stdout
        if stdout is None:
            raise OpenPIFastWorkerProtocolError("worker stdout is unavailable")

        try:
            line = await asyncio.wait_for(stdout.readline(), timeout=timeout_s)
        except TimeoutError as exc:
            raise OpenPIFastWorkerProtocolError("worker reply timed out") from exc

        if not line:
            stderr = await self._stderr_text()
            raise OpenPIFastWorkerProtocolError(
                f"worker exited before replying: returncode={self._process.returncode} stderr={stderr!r}"
            )

        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenPIFastWorkerProtocolError(
                f"worker emitted invalid JSON: {line.decode('utf-8', errors='replace').rstrip()!r}"
            ) from exc

    async def _await_ready(self) -> None:
        message = await self._read_message(self._spec.startup_timeout_s)
        if message.get("event") != "ready":
            raise OpenPIFastWorkerProtocolError(f"worker handshake missing ready event: {message!r}")
        if int(message.get("protocol_version", -1)) != OPENPI_FAST_WORKER_PROTOCOL_VERSION:
            raise OpenPIFastWorkerProtocolError(
                "worker protocol version mismatch: "
                f"expected {OPENPI_FAST_WORKER_PROTOCOL_VERSION}, got {message.get('protocol_version')!r}"
            )

    async def _stderr_text(self) -> str:
        stderr = self._process.stderr
        if stderr is None:
            return ""
        if self._process.returncode is None:
            return ""
        try:
            data = await asyncio.wait_for(stderr.read(), timeout=0.1)
        except Exception:
            return ""
        return data.decode("utf-8", errors="replace").strip()

    def timeout_for(self, op: str) -> float:
        if op == "create_session":
            return self._spec.create_session_timeout_s
        if op == "save_weights":
            return self._spec.save_weights_timeout_s
        if op == "load_weights":
            return self._spec.load_weights_timeout_s
        return self._spec.request_timeout_s

    async def request(
        self,
        op: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise OpenPIFastWorkerProtocolError("worker client is closed")

        stdin = self._process.stdin
        if stdin is None:
            raise OpenPIFastWorkerProtocolError("worker stdin is unavailable")

        async with self._request_lock:
            request_id = str(next(self._request_ids))
            stdin.write(
                (
                    json.dumps({"id": request_id, "op": op, "payload": payload or {}})
                    + "\n"
                ).encode("utf-8")
            )
            await stdin.drain()

            effective_timeout = self.timeout_for(op) if timeout_s is None else float(timeout_s)
            deadline = time.monotonic() + effective_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenPIFastWorkerProtocolError(
                        f"worker op {op!r} timed out or returned invalid protocol after {effective_timeout}s: worker reply timed out"
                    )
                try:
                    message = await self._read_message(remaining)
                except OpenPIFastWorkerProtocolError as exc:
                    raise OpenPIFastWorkerProtocolError(
                        f"worker op {op!r} timed out or returned invalid protocol after {effective_timeout}s: {exc}"
                    ) from exc
                if message.get("id") == request_id:
                    break
                raise OpenPIFastWorkerProtocolError(
                    f"worker reply id mismatch for op {op!r}: expected {request_id!r}, got {message.get('id')!r}"
                )
            if message.get("ok") is not True:
                error = message.get("error") or {}
                raise OpenPIFastWorkerRemoteError(
                    error_type=str(error.get("type") or "RuntimeError"),
                    message=str(error.get("message") or "worker request failed"),
                    traceback_text=error.get("traceback"),
                )
            payload_out = message.get("payload")
            if payload_out is None:
                return {}
            if not isinstance(payload_out, dict):
                raise OpenPIFastWorkerProtocolError(
                    f"worker payload must be a dict, got {type(payload_out).__name__}"
                )
            return payload_out

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._process.returncode is not None:
            return

        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()
