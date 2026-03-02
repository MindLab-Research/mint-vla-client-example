import os
import tarfile
import tempfile
from urllib.parse import urlparse

import requests
import tinker


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")


def _download_to_tmp(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        full = url
    else:
        full = f"{BASE_URL}{url}"
    r = requests.get(full, timeout=300)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(prefix="ckpt_", suffix=".tar.gz")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _assert_is_tar_gz(path: str) -> None:
    with tarfile.open(path, "r:gz") as tf:
        members = tf.getmembers()
    if not members:
        raise RuntimeError(f"Empty archive: {path}")


def main() -> int:
    os.environ.setdefault("TINKER_BASE_URL", BASE_URL)
    os.environ.setdefault("TINKER_API_KEY", API_KEY)

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    training_client = service_client.create_lora_training_client(
        base_model="Qwen/Qwen3-0.6B",
        rank=8,
    )

    model_id = training_client.model_id
    try:
        # Use distinct names: training and sampler checkpoints are different classes.
        training_path = training_client.save_state("w0001").result().path
        sampler_path = training_client.save_weights_for_sampler("s0001").result().path

        # Path-based download: must work for canonical tinker paths.
        training_url = rest_client.get_checkpoint_archive_url_from_tinker_path(training_path).result()
        sampler_url = rest_client.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()

        training_archive = _download_to_tmp(training_url.url)
        sampler_archive = _download_to_tmp(sampler_url.url)
        _assert_is_tar_gz(training_archive)
        _assert_is_tar_gz(sampler_archive)

        # ID-based download: checkpoint_id must accept "weights/<id>" and "sampler_weights/<id>".
        parsed_training = tinker.types.checkpoint.ParsedCheckpointTinkerPath.from_tinker_path(training_path)
        parsed_sampler = tinker.types.checkpoint.ParsedCheckpointTinkerPath.from_tinker_path(sampler_path)

        training_url2 = rest_client.get_checkpoint_archive_url(parsed_training.training_run_id, parsed_training.checkpoint_id).result()
        sampler_url2 = rest_client.get_checkpoint_archive_url(parsed_sampler.training_run_id, parsed_sampler.checkpoint_id).result()

        training_archive2 = _download_to_tmp(training_url2.url)
        sampler_archive2 = _download_to_tmp(sampler_url2.url)
        _assert_is_tar_gz(training_archive2)
        _assert_is_tar_gz(sampler_archive2)

        # Invalid checkpoint should be stable 4xx (not protocol-shape mismatch).
        try:
            rest_client.get_checkpoint_archive_url(model_id, "weights/does-not-exist").result()
        except Exception as e:
            # SDK raises APIStatusError on 4xx (acceptable).
            if "Expected a redirect response" in str(e):
                raise RuntimeError(f"Protocol mismatch for 4xx: {e}") from e

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
