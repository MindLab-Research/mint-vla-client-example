import io
import os
import sys
import tarfile
import tempfile
import time
import base64
import hashlib
import json
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL")
if not BASE_URL:
    port = os.environ.get("MINT_PORT", "8000")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

ADMIN_API_KEY = os.environ.get("MINT_ADMIN_API_KEY") or os.environ.get("MINT_API_KEY", "")

TOKEN_SECRET_KEY = os.environ.get("MINT_TOKEN_SECRET_KEY") or os.environ.get("MINT_USER_TOKEN_SECRET_KEY") or ""
USER_API_KEY = os.environ.get("MINT_USER_API_KEY", "").strip()
USER_ID = os.environ.get("MINT_USER_ID", "issue_149_user")

MODEL = os.environ.get("MINT_MODEL", "Qwen/Qwen3-0.6B")
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "16"))

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "120"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "1.0"))

GENERIC_ERROR_MESSAGE = "Operation failed. Contact administrator if issue persists."


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def _post_json(path: str, body: dict[str, Any], *, api_key: str, timeout_s: float = 60.0) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, headers=_headers(api_key), json=body, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]!r}")
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(out)}")
    return out


def _poll_future(request_id: str, *, api_key: str) -> dict[str, Any]:
    t0 = time.time()
    while True:
        r = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(api_key),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if r.status_code == 408:
            if time.time() - t0 > POLL_TIMEOUT_S:
                raise RuntimeError(f"retrieve_future timeout after {POLL_TIMEOUT_S:.1f}s request_id={request_id}")
            time.sleep(POLL_SLEEP_S)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {r.status_code}: {r.text[:500]!r}")
        out = r.json()
        if not isinstance(out, dict):
            raise RuntimeError(f"retrieve_future returned non-dict json: {type(out)}")
        return out


def _assert_auth_enabled() -> None:
    r = requests.get(f"{BASE_URL}/api/v1/get_server_capabilities", timeout=10.0)
    if r.status_code == 200:
        raise RuntimeError(
            "auth appears disabled (GET /api/v1/get_server_capabilities returned 200 without API key). "
            "Start the issue-scoped server with MINT_API_KEY (admin) and MINT_TOKEN_SECRET_KEY (for sk- tokens)."
        )
    if r.status_code != 401:
        raise RuntimeError(f"unexpected auth probe status={r.status_code} body={r.text[:200]!r}")


def _get_user_api_key() -> str:
    if USER_API_KEY:
        return USER_API_KEY
    if not TOKEN_SECRET_KEY:
        raise RuntimeError(
            "Need non-admin credentials: set MINT_USER_API_KEY to a valid sk- token, "
            "or set MINT_TOKEN_SECRET_KEY (matching the server) so this script can mint one."
        )
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except Exception as e:
        raise RuntimeError(f"cryptography import failed: {e!r}")

    key = hashlib.sha256(TOKEN_SECRET_KEY.encode()).digest()
    iv = os.urandom(16)

    payload = json.dumps({"user_id": USER_ID}, ensure_ascii=True, separators=(",", ":")).encode()
    pad_len = 16 - (len(payload) % 16)
    padded = payload + bytes([pad_len] * pad_len)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()

    token_body = base64.urlsafe_b64encode(iv + encrypted).decode().rstrip("=")
    return f"sk-mint-{token_body}"


def _build_minimal_ckpt_tar_gz() -> str:
    fd, out_path = tempfile.mkstemp(prefix="issue_149_ckpt_", suffix=".tar.gz")
    os.close(fd)
    root = "checkpoint"
    with tarfile.open(out_path, "w:gz") as tf:
        for rel, content in (
            ("adapter_model.safetensors", b"not-a-real-safetensors"),
            ("optimizer.pt", b"not-a-real-optimizer"),
        ):
            ti = tarfile.TarInfo(name=f"{root}/{rel}")
            ti.size = len(content)
            ti.mtime = int(time.time())
            tf.addfile(ti, fileobj=io.BytesIO(content))
    return out_path


def _upload_ckpt(archive_path: str, *, api_key: str) -> dict[str, Any]:
    with open(archive_path, "rb") as f:
        files = {"file": (os.path.basename(archive_path), f, "application/gzip")}
        r = requests.post(
            f"{BASE_URL}/api/v1/checkpoints/upload",
            headers=_headers(api_key),
            files=files,
            timeout=120.0,
        )
    if r.status_code != 200:
        raise RuntimeError(f"POST /api/v1/checkpoints/upload -> {r.status_code}: {r.text[:500]!r}")
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"upload returned non-dict json: {type(out)}")
    return out


def main() -> int:
    archive_path: str | None = None
    try:
        if not ADMIN_API_KEY:
            return _fail("missing admin key: set MINT_API_KEY or MINT_ADMIN_API_KEY")

        _assert_auth_enabled()
        user_key = _get_user_api_key()

        archive_path = _build_minimal_ckpt_tar_gz()
        uploaded = _upload_ckpt(archive_path, api_key=ADMIN_API_KEY)
        ckpt_id = uploaded.get("checkpoint_id")
        if not isinstance(ckpt_id, str) or not ckpt_id.startswith("ckpt_"):
            return _fail(f"upload returned invalid checkpoint_id={ckpt_id!r}")

        session_id = f"issue-149-{int(time.time())}"
        create = _post_json(
            "/api/v1/create_model_from_state",
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": MODEL,
                "state_path": ckpt_id,
                "lora_config": {"rank": LORA_RANK},
                "load_optimizer": False,
                "user_metadata": {"issue": 149},
            },
            api_key=user_key,
            timeout_s=30.0,
        )
        request_id = create.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"create_model_from_state returned invalid request_id={request_id!r}")

        out = _poll_future(request_id, api_key=user_key)
        err = out.get("error")
        if not isinstance(err, str) or not err:
            return _fail(f"expected create_model_from_state to fail, got: {out!r}")

        if err == GENERIC_ERROR_MESSAGE:
            return _fail(
                "error is masked for non-admin user; expected explicit permission error (e.g. 'Access denied')"
            )
        if not err.startswith("Access denied"):
            return _fail(f"expected permission error prefix 'Access denied', got error={err!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if archive_path:
            try:
                os.unlink(archive_path)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
