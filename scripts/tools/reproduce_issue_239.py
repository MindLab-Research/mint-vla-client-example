import os
import tarfile
import tempfile

import requests
import tinker


BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")


def _download(url: str) -> str:
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(prefix="ckpt_", suffix=".tar.gz")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _upload(path: str) -> str:
    with open(path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/v1/checkpoints/upload",
            files={"file": ("archive.tar.gz", f, "application/gzip")},
            headers={"X-API-Key": API_KEY},
            timeout=600,
        )
    resp.raise_for_status()
    payload = resp.json()
    ckpt_id = payload.get("checkpoint_id")
    if not isinstance(ckpt_id, str) or not ckpt_id:
        raise RuntimeError(f"upload response missing checkpoint_id: {payload!r}")
    return ckpt_id


def _assert_is_tar_gz(path: str) -> None:
    with tarfile.open(path, "r:gz") as tf:
        if not tf.getmembers():
            raise RuntimeError(f"Empty archive: {path}")


def main() -> int:
    os.environ.setdefault("MINT_BASE_URL", BASE_URL)
    os.environ.setdefault("MINT_API_KEY", API_KEY)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    tc = sc.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)
    model_id = tc.model_id

    try:
        # 1) Sampler checkpoint loop: download -> upload -> resume(without optimizer), reject(with optimizer)
        sampler_path = tc.save_weights_for_sampler("sampler-final").result().path
        sampler_signed = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
        sampler_archive = _download(sampler_signed.url)
        _assert_is_tar_gz(sampler_archive)
        sampler_uploaded = _upload(sampler_archive)

        # Must reject optimizer restore for sampler checkpoints.
        try:
            tc.load_state_with_optimizer(sampler_uploaded).result()
            raise RuntimeError("expected load_state_with_optimizer(sampler_uploaded) to fail")
        except Exception:
            pass

        # Must allow non-optimizer load.
        tc.load_state(sampler_uploaded).result()

        # 2) Training checkpoint loop: download -> upload -> resume(with optimizer)
        training_path = tc.save_state("weights-final").result().path
        training_signed = rc.get_checkpoint_archive_url_from_tinker_path(training_path).result()
        training_archive = _download(training_signed.url)
        _assert_is_tar_gz(training_archive)
        training_uploaded = _upload(training_archive)

        tc.load_state_with_optimizer(training_uploaded).result()

        print("PASS", flush=True)
        return 0
    finally:
        try:
            requests.delete(
                f"{BASE_URL}/api/v1/models/{model_id}",
                headers={"X-API-Key": API_KEY},
                timeout=60,
            )
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
