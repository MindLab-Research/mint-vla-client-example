#!/usr/bin/env python3
import asyncio
import os
import time
import uuid

import httpx


BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _sample_payload(session_id: str) -> dict:
    return {
        "model_id": session_id,
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
        "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
    }


async def _post_json(client: httpx.AsyncClient, path: str, payload: dict) -> httpx.Response:
    return await client.post(f"{BASE_URL}{path}", json=payload, headers=_headers(), timeout=30.0)


async def main() -> None:
    trials = int(os.environ.get("MINT_REPRO_TRIALS", "50"))
    poll_attempts = int(os.environ.get("MINT_REPRO_POLL_ATTEMPTS", "20"))
    poll_sleep_s = float(os.environ.get("MINT_REPRO_POLL_SLEEP_S", "0.05"))

    limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
    transport = httpx.AsyncHTTPTransport(retries=0)

    print(f"BASE_URL={BASE_URL}")
    print(f"trials={trials} poll_attempts={poll_attempts} poll_sleep_s={poll_sleep_s}")

    async with httpx.AsyncClient(http2=False, limits=limits, transport=transport) as create_client, httpx.AsyncClient(
        http2=False, limits=limits, transport=transport
    ) as poll_client:
        for i in range(trials):
            session_id = f"repro_nonexistent_session_{uuid.uuid4()}"
            t0 = time.time()
            create_r = await _post_json(create_client, "/api/v1/asample", _sample_payload(session_id))
            create_dt_ms = int((time.time() - t0) * 1000)
            create_r.raise_for_status()
            request_id = create_r.json()["request_id"]
            print(f"[{i}] asample request_id={request_id} dt_ms={create_dt_ms}")

            for j in range(poll_attempts):
                poll_r = await _post_json(poll_client, "/api/v1/retrieve_future", {"request_id": request_id})
                if poll_r.status_code == 404:
                    print(f"[{i}] retrieve_future 404 attempt={j} body={poll_r.text.strip()}")
                    raise SystemExit(1)
                if poll_r.status_code == 408:
                    await asyncio.sleep(poll_sleep_s)
                    continue
                poll_r.raise_for_status()
                body = poll_r.json()
                if isinstance(body, dict) and "error" in body:
                    print(f"[{i}] retrieve_future 200 error={body.get('error')!r}")
                    break
                print(f"[{i}] retrieve_future 200 body_keys={sorted(body.keys()) if isinstance(body, dict) else type(body)}")
                break

    print("PASS: no retrieve_future 404 observed")


if __name__ == "__main__":
    asyncio.run(main())
