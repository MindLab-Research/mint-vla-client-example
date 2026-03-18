from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OPENPI_FAST_WORKER_PROTOCOL_VERSION = 1


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
    mint_root = Path(__file__).resolve().parents[2]
    repo_root = mint_root.parents[1]
    openpi_src = repo_root / "src" / "openpi" / "src"

    entries = [str(mint_root)]
    if openpi_src.exists():
        entries.append(str(openpi_src))
    return tuple(entries)


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
    worker_module: str = "tinker_server.backend.openpi_fast_worker"
    pythonpath: tuple[str, ...] = field(default_factory=_default_pythonpath)
    startup_timeout_s: float = 30.0
    request_timeout_s: float = 300.0
    cwd: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "OpenPIFastRuntimeSpec":
        pythonpath_env = os.environ.get("MINT_OPENPI_FAST_PYTHONPATH", "").strip()
        pythonpath = tuple(s for s in pythonpath_env.split(os.pathsep) if s) or _default_pythonpath()
        return cls(
            python_executable=os.environ.get("MINT_OPENPI_FAST_PYTHON", sys.executable),
            worker_module=os.environ.get(
                "MINT_OPENPI_FAST_WORKER_MODULE",
                "tinker_server.backend.openpi_fast_worker",
            ),
            pythonpath=pythonpath,
            startup_timeout_s=float(os.environ.get("MINT_OPENPI_FAST_STARTUP_TIMEOUT_S", "30")),
            request_timeout_s=float(os.environ.get("MINT_OPENPI_FAST_REQUEST_TIMEOUT_S", "300")),
            cwd=os.environ.get("MINT_OPENPI_FAST_CWD") or None,
        )

    def build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = _merge_pythonpath(self.pythonpath, env.get("PYTHONPATH"))
        env.update(self.extra_env)
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
            stderr=asyncio.subprocess.PIPE,
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

    async def request(self, op: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
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

            message = await self._read_message(self._spec.request_timeout_s)
            if message.get("id") != request_id:
                raise OpenPIFastWorkerProtocolError(
                    f"worker reply id mismatch: expected {request_id!r}, got {message.get('id')!r}"
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
