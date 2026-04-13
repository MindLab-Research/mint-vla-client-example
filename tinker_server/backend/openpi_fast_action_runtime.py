from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .openpi_fast_runtime import OpenPIFastRuntimeSpec, OpenPIFastWorkerClient


def find_openpi_policy_checkpoint_dir(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "params").is_dir():
        return root

    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        if not (child / "params").is_dir():
            continue
        candidates.append((int(child.name), child))
    if not candidates:
        raise FileNotFoundError(f"OpenPI policy checkpoint missing params directory: {root}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


@dataclass(frozen=True)
class OpenPIFastActionRuntimeSpec(OpenPIFastRuntimeSpec):
    worker_module: str = "tinker_server.backend.openpi_fast_action_worker"

    @classmethod
    def from_env(cls) -> "OpenPIFastActionRuntimeSpec":
        base = OpenPIFastRuntimeSpec.from_env()
        request_timeout_s = float(
            os.environ.get("MINT_OPENPI_FAST_ACTION_REQUEST_TIMEOUT_S", str(base.request_timeout_s))
        )
        create_session_timeout_s = float(
            os.environ.get(
                "MINT_OPENPI_FAST_ACTION_CREATE_SESSION_TIMEOUT_S",
                str(base.create_session_timeout_s),
            )
        )
        return cls(
            python_executable=os.environ.get("MINT_OPENPI_FAST_ACTION_PYTHON", "").strip() or base.python_executable,
            worker_module=os.environ.get(
                "MINT_OPENPI_FAST_ACTION_WORKER_MODULE",
                "tinker_server.backend.openpi_fast_action_worker",
            ),
            pythonpath=base.pythonpath,
            startup_timeout_s=float(
                os.environ.get("MINT_OPENPI_FAST_ACTION_STARTUP_TIMEOUT_S", str(base.startup_timeout_s))
            ),
            request_timeout_s=request_timeout_s,
            create_session_timeout_s=create_session_timeout_s,
            save_weights_timeout_s=base.save_weights_timeout_s,
            load_weights_timeout_s=base.load_weights_timeout_s,
            cwd=os.environ.get("MINT_OPENPI_FAST_ACTION_CWD") or base.cwd,
            extra_env=base.extra_env,
        )


class OpenPIFastActionWorkerClient(OpenPIFastWorkerClient):
    @classmethod
    async def start(
        cls,
        spec: OpenPIFastActionRuntimeSpec | None = None,
    ) -> "OpenPIFastActionWorkerClient":
        return await super().start(spec or OpenPIFastActionRuntimeSpec.from_env())
