import io
import json
import os
import subprocess
import sys
import tarfile
import uuid

import requests


GATEWAY_BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:10276").rstrip("/")
UPSTREAM_BASE_URL = os.environ.get("TINKER_UPSTREAM_BASE_URL", "http://localhost:10277").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
BASE_MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-0.6B")
MINT_DEV_HOST = os.environ.get("MINT_DEV_HOST", "mint-dev")
RAY_ADDRESS = os.environ.get("TINKER_RAY_ADDRESS", "192.168.37.63:6379")
GATEWAY_NAMESPACE = os.environ.get("TINKER_GATEWAY_NAMESPACE", "tinker_yiwen_issue_276_gateway")
UPSTREAM_ALIAS = os.environ.get("TINKER_UPSTREAM_ALIAS", "issue276-upstream")
CHECKPOINTS_DIR = os.environ.get("TINKER_CHECKPOINT_DIR", "/vePFS-Mindverse/share/tinker_checkpoints")
HTTP_TIMEOUT_S = float(os.environ.get("TINKER_HTTP_TIMEOUT_S", "60"))


def _headers(*, user_agent: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    if user_agent:
        headers["User-Agent"] = user_agent
    return headers


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _make_training_archive(*, root: str, model_id: str) -> bytes:
    metadata = {
        "checkpoint_id": root,
        "owner_id": None,
        "model_id": model_id,
        "model_name": BASE_MODEL,
        "created_at": "2026-03-08T00:00:00Z",
        "step": 0,
        "checkpoint_type": "training",
        "optimizer_present": True,
        "backend": "dense",
        "type": "training",
    }
    training_meta = {"current_step": 0, "learning_rate": 1e-4}

    files = {
        "adapter_model.safetensors": b"dummy-lora",
        "optimizer.pt": b"dummy-optimizer",
        "adapter_config.json": json.dumps({"base_model_name_or_path": BASE_MODEL}).encode("utf-8"),
        "training_meta.json": json.dumps(training_meta).encode("utf-8"),
        "metadata.json": json.dumps(metadata).encode("utf-8"),
    }

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel_path, data in files.items():
            name = f"{root}/{rel_path}"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _seed_gateway_remote_training_model(*, model_id: str) -> None:
    script = f"""
cd /root/tinker_project/tinker-server-issue-276
env RAY_ADDRESS={RAY_ADDRESS} \
PYTHONPATH=/root/tinker_project/tinker-server-issue-276 \
TINKER_RAY_NAMESPACE={GATEWAY_NAMESPACE} \
MINT_RAY_NAMESPACE={GATEWAY_NAMESPACE} \
/root/venv_k2_py31213/bin/python - <<'PY'
from tinker_server.backend import gateway_session_store
gateway_session_store.upsert_training_model(
    model_id={model_id!r},
    upstream_alias={UPSTREAM_ALIAS!r},
    base_model={BASE_MODEL!r},
)
print("seeded")
PY
"""
    proc = subprocess.run(
        ["ssh", MINT_DEV_HOST, script],
        capture_output=True,
        text=True,
        timeout=HTTP_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to seed gateway_session_store rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    if "seeded" not in proc.stdout:
        raise RuntimeError(f"gateway_session_store seed missing confirmation: {proc.stdout!r}")


def _promote_uploaded_checkpoint(*, model_id: str, checkpoint_id: str) -> None:
    proc = subprocess.run(
        [
            "ssh",
            MINT_DEV_HOST,
            (
                f"mkdir -p {CHECKPOINTS_DIR}/anonymous/{model_id} && "
                f"rm -rf {CHECKPOINTS_DIR}/anonymous/{model_id}/{checkpoint_id} && "
                f"mv {CHECKPOINTS_DIR}/anonymous/{checkpoint_id} {CHECKPOINTS_DIR}/anonymous/{model_id}/{checkpoint_id}"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=HTTP_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to place uploaded checkpoint under model root rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )


def _upload_training_checkpoint(*, model_id: str) -> str:
    archive_root = f"checkpoint-{uuid.uuid4().hex[:8]}"
    payload = _make_training_archive(root=archive_root, model_id=model_id)
    resp = requests.post(
        f"{UPSTREAM_BASE_URL}/api/v1/checkpoints/upload",
        headers=_headers(),
        files={"file": ("checkpoint.tar.gz", payload, "application/gzip")},
        timeout=max(HTTP_TIMEOUT_S, 300.0),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"upstream checkpoint upload returned {resp.status_code}: {resp.text[:500]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"upstream checkpoint upload returned non-dict json: {type(data)}")
    ckpt_id = data.get("checkpoint_id")
    if not isinstance(ckpt_id, str) or not ckpt_id:
        raise RuntimeError(f"upstream checkpoint upload missing checkpoint_id: {data!r}")
    return ckpt_id


def main() -> int:
    model_id = f"issue276_{uuid.uuid4().hex[:10]}"
    try:
        for base_url, label in ((UPSTREAM_BASE_URL, "upstream"), (GATEWAY_BASE_URL, "gateway")):
            resp = requests.get(f"{base_url}/api/v1/healthz", headers=_headers(), timeout=10)
            if resp.status_code != 200:
                return _fail(f"{label} healthz returned {resp.status_code}: {resp.text[:500]!r}")

        checkpoint_id = _upload_training_checkpoint(model_id=model_id)
        _promote_uploaded_checkpoint(model_id=model_id, checkpoint_id=checkpoint_id)
        _seed_gateway_remote_training_model(model_id=model_id)

        listed_resp = requests.get(
            f"{GATEWAY_BASE_URL}/api/v1/training_runs/{model_id}/checkpoints",
            headers=_headers(),
            timeout=HTTP_TIMEOUT_S,
        )
        if listed_resp.status_code != 200:
            return _fail(
                f"gateway list_checkpoints returned {listed_resp.status_code}: {listed_resp.text[:500]!r}"
            )
        listed = listed_resp.json()
        checkpoints = listed.get("checkpoints")
        if not isinstance(checkpoints, list):
            return _fail(f"list_checkpoints returned unexpected payload: {listed!r}")
        checkpoint_ids = {ckpt.get("checkpoint_id") for ckpt in checkpoints if isinstance(ckpt, dict)}
        expected_checkpoint_id = f"weights/{checkpoint_id}"
        if expected_checkpoint_id not in checkpoint_ids:
            return _fail(f"list_checkpoints missing {expected_checkpoint_id!r}: {listed!r}")

        archive_resp = requests.get(
            f"{GATEWAY_BASE_URL}/api/v1/training_runs/{model_id}/checkpoints/{expected_checkpoint_id}/archive",
            headers=_headers(user_agent="AsyncTinker/Python 0.13.1"),
            timeout=HTTP_TIMEOUT_S,
            allow_redirects=False,
        )
        if archive_resp.status_code != 302:
            return _fail(
                f"gateway archive route returned {archive_resp.status_code}, expected 302: {archive_resp.text[:500]!r}"
            )
        if not archive_resp.headers.get("Location"):
            return _fail(f"archive redirect missing Location header: {dict(archive_resp.headers)!r}")

        direct = requests.get(
            f"{GATEWAY_BASE_URL}/api/v1/training_runs/{model_id}/checkpoints/{expected_checkpoint_id}/archive",
            headers=_headers(),
            params={"direct": "1"},
            timeout=max(HTTP_TIMEOUT_S, 300.0),
            stream=True,
        )
        if direct.status_code != 200:
            return _fail(f"direct archive download returned {direct.status_code}: {direct.text[:500]!r}")
        first_chunk = next(direct.iter_content(chunk_size=65536), b"")
        direct.close()
        if not first_chunk.startswith(b"\x1f\x8b"):
            return _fail(f"archive payload missing gzip header: {first_chunk[:16]!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
