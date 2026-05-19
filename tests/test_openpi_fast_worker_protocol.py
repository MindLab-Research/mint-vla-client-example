from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).resolve().parents[1]


def _runtime_spec():
    from mint_server.backend.openpi_fast_runtime import OpenPIFastRuntimeSpec

    return OpenPIFastRuntimeSpec(
        python_executable=sys.executable,
        worker_module="tests.openpi_fast_dummy_worker",
        pythonpath=(str(SRC_ROOT),),
    )


def test_openpi_fast_worker_client_round_trips_json_payload() -> None:
    from mint_server.backend.openpi_fast_runtime import OpenPIFastWorkerClient

    async def _run() -> None:
        client = await OpenPIFastWorkerClient.start(_runtime_spec())
        try:
            payload = await client.request("echo", {"value": 3})
            assert payload == {"value": 3}
        finally:
            await client.close()

    asyncio.run(_run())


def test_openpi_fast_worker_client_surfaces_remote_errors() -> None:
    from mint_server.backend.openpi_fast_runtime import (
        OpenPIFastWorkerClient,
        OpenPIFastWorkerRemoteError,
    )

    async def _run() -> None:
        client = await OpenPIFastWorkerClient.start(_runtime_spec())
        try:
            with pytest.raises(OpenPIFastWorkerRemoteError, match="boom"):
                await client.request("fail", {"message": "boom"})
        finally:
            await client.close()

    asyncio.run(_run())


def test_openpi_fast_worker_client_rejects_mismatched_reply_ids() -> None:
    from mint_server.backend.openpi_fast_runtime import (
        OpenPIFastWorkerClient,
        OpenPIFastWorkerProtocolError,
    )

    async def _run() -> None:
        client = await OpenPIFastWorkerClient.start(_runtime_spec())
        try:
            with pytest.raises(OpenPIFastWorkerProtocolError, match="reply id"):
                await client.request("mismatch", {"value": 7})
        finally:
            await client.close()

    asyncio.run(_run())


def test_openpi_fast_worker_client_rejects_malformed_replies() -> None:
    from mint_server.backend.openpi_fast_runtime import (
        OpenPIFastWorkerClient,
        OpenPIFastWorkerProtocolError,
    )

    async def _run() -> None:
        client = await OpenPIFastWorkerClient.start(_runtime_spec())
        try:
            with pytest.raises(OpenPIFastWorkerProtocolError, match="JSON"):
                await client.request("malformed", {"value": 9})
        finally:
            await client.close()

    asyncio.run(_run())


def test_openpi_fast_worker_client_supports_per_request_timeouts() -> None:
    from mint_server.backend.openpi_fast_runtime import (
        OpenPIFastWorkerClient,
        OpenPIFastWorkerProtocolError,
    )

    async def _run() -> None:
        client = await OpenPIFastWorkerClient.start(_runtime_spec())
        try:
            with pytest.raises(OpenPIFastWorkerProtocolError, match="sleep"):
                await client.request("sleep", {"seconds": 0.2}, timeout_s=0.01)
        finally:
            await client.close()

    asyncio.run(_run())
