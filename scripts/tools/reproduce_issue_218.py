import os
import sys
import tempfile
import time
import uuid

import requests

GATEWAY_BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
UPSTREAM_BASE_URL = os.environ.get("TINKER_UPSTREAM_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

BASE_MODEL = os.environ.get("TINKER_BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

HTTP_TIMEOUT_S = float(os.environ.get("TINKER_HTTP_TIMEOUT_S", "60"))
FUTURE_TIMEOUT_S = float(os.environ.get("TINKER_FUTURE_TIMEOUT_S", "1800"))
POLL_INTERVAL_S = float(os.environ.get("TINKER_POLL_INTERVAL_S", "2.0"))


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


def _post_json(base_url: str, path: str, payload: dict, *, timeout_s: float) -> requests.Response:
    url = f"{base_url}{path}"
    return requests.post(url, json=payload, headers=_headers(), timeout=timeout_s)


def _get_stream(base_url: str, path: str, *, timeout_s: float) -> requests.Response:
    url = f"{base_url}{path}"
    return requests.get(url, headers=_headers(), timeout=timeout_s, stream=True)


def _wait_future(*, request_id: str, label: str) -> dict:
    start = time.time()
    last_status = None
    while time.time() - start < FUTURE_TIMEOUT_S:
        resp = _post_json(
            GATEWAY_BASE_URL,
            "/api/v1/retrieve_future",
            {"request_id": request_id},
            timeout_s=HTTP_TIMEOUT_S,
        )
        last_status = resp.status_code
        if resp.status_code == 408:
            time.sleep(POLL_INTERVAL_S)
            continue
        if resp.status_code != 200:
            raise RuntimeError(
                f"{label} request_id={request_id!r} expected 200/408, got {resp.status_code}: {resp.text}"
            )
        try:
            return resp.json()
        except Exception as e:
            raise RuntimeError(
                f"{label} request_id={request_id!r} returned non-JSON: {resp.text}"
            ) from e
    raise TimeoutError(f"timeout waiting for {label} request_id={request_id!r} last_status={last_status}")


def _parse_checkpoint_dirname(uri: str) -> str:
    # save_state returns mint://{model_id}/{checkpoint_name} or tinker://{model_id}/{checkpoint_name}
    parts = uri.split("/")
    if len(parts) < 2:
        raise ValueError(f"unexpected checkpoint uri: {uri!r}")
    checkpoint_name = parts[-1]
    if not checkpoint_name:
        raise ValueError(f"unexpected checkpoint uri: {uri!r}")
    return checkpoint_name


def _download_to_tmpfile(*, model_id: str, checkpoint_id: str) -> str:
    resp = _get_stream(
        UPSTREAM_BASE_URL,
        f"/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive",
        timeout_s=max(HTTP_TIMEOUT_S, 300.0),
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"download archive expected 200, got {resp.status_code}: {resp.text}"
        )

    fd, tmp_path = tempfile.mkstemp(prefix="repro_issue_218_", suffix=".tar.gz")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
    finally:
        resp.close()
    return tmp_path


def _upload_archive_to_gateway(archive_path: str) -> str:
    url = f"{GATEWAY_BASE_URL}/api/v1/checkpoints/upload"
    with open(archive_path, "rb") as f:
        files = {"file": (os.path.basename(archive_path), f, "application/gzip")}
        resp = requests.post(
            url,
            headers=_headers(),
            files=files,
            timeout=max(HTTP_TIMEOUT_S, 300.0),
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"upload archive expected 200, got {resp.status_code}: {resp.text}"
        )
    body = resp.json()
    ckpt_id = body.get("checkpoint_id")
    if not isinstance(ckpt_id, str) or not ckpt_id:
        raise RuntimeError(f"upload archive returned invalid checkpoint_id: {body!r}")
    return ckpt_id


def main() -> int:
    # 1) Create a remote training model (gateway-routed to upstream).
    resp = _post_json(
        GATEWAY_BASE_URL,
        "/api/v1/create_session",
        {"tags": ["scripts/tools/reproduce_issue_218.py"], "user_metadata": {}, "sdk_version": "repro_issue_218"},
        timeout_s=HTTP_TIMEOUT_S,
    )
    if resp.status_code != 200:
        return _fail(f"create_session expected 200, got {resp.status_code}: {resp.text}")
    session_id = resp.json().get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return _fail(f"create_session missing session_id: {resp.text}")

    model_seq_id = 0
    resp = _post_json(
        GATEWAY_BASE_URL,
        "/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": 8},
        },
        timeout_s=HTTP_TIMEOUT_S,
    )
    if resp.status_code != 200:
        return _fail(f"create_model expected 200, got {resp.status_code}: {resp.text}")
    create_req_id = resp.json().get("request_id")
    if not isinstance(create_req_id, str) or not create_req_id:
        return _fail(f"create_model missing request_id: {resp.text}")

    create_out = _wait_future(request_id=create_req_id, label="create_model")
    model_id = create_out.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        return _fail(f"create_model future missing model_id: {create_out!r}")

    # 2) Save a real checkpoint on the upstream (via gateway).
    ckpt_name = f"repro-218-{uuid.uuid4().hex[:10]}"
    resp = _post_json(
        GATEWAY_BASE_URL,
        "/api/v1/save_state",
        {"model_id": model_id, "path": ckpt_name},
        timeout_s=HTTP_TIMEOUT_S,
    )
    if resp.status_code != 200:
        return _fail(f"save_state expected 200, got {resp.status_code}: {resp.text}")
    save_req_id = resp.json().get("request_id")
    if not isinstance(save_req_id, str) or not save_req_id:
        return _fail(f"save_state missing request_id: {resp.text}")
    save_out = _wait_future(request_id=save_req_id, label="save_state")
    saved_uri = save_out.get("path")
    if not isinstance(saved_uri, str) or not saved_uri:
        return _fail(f"save_state future missing path: {save_out!r}")

    upstream_checkpoint_id = _parse_checkpoint_dirname(saved_uri)

    # 3) Copy the upstream checkpoint into the gateway checkpoint store.
    archive_path = _download_to_tmpfile(model_id=model_id, checkpoint_id=upstream_checkpoint_id)
    try:
        gateway_ckpt_id = _upload_archive_to_gateway(archive_path)
    finally:
        try:
            os.unlink(archive_path)
        except OSError:
            pass

    # 4) Cross-cluster resume: create_model_from_state for a gateway-routed model using a gateway-local checkpoint id.
    # Expected behavior (after fix): gateway proxies checkpoint to upstream, then upstream create_model_from_state succeeds.
    resp = _post_json(
        GATEWAY_BASE_URL,
        "/api/v1/create_model_from_state",
        {
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": BASE_MODEL,
            "state_path": gateway_ckpt_id,
            "load_optimizer": True,
            "lora_config": {"rank": 8},
        },
        timeout_s=HTTP_TIMEOUT_S,
    )
    if resp.status_code != 200:
        return _fail(
            f"create_model_from_state expected 200, got {resp.status_code}: {resp.text}"
        )
    cms_req_id = resp.json().get("request_id")
    if not isinstance(cms_req_id, str) or not cms_req_id:
        return _fail(f"create_model_from_state missing request_id: {resp.text}")

    cms_out = _wait_future(request_id=cms_req_id, label="create_model_from_state")
    if cms_out.get("type") != "create_model_from_state" or not cms_out.get("model_id"):
        return _fail(f"create_model_from_state future unexpected payload: {cms_out!r}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
