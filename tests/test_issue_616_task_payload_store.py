from __future__ import annotations

from pydantic import BaseModel
import pytest

from tinker_server.backend.task_payload_store import TaskPayloadStore, TaskPayloadStoreError


class _PayloadModel(BaseModel):
    request_id: str
    values: list[int]


def test_task_payload_store_writes_and_reads_json_payload(tmp_path) -> None:
    store = TaskPayloadStore(tmp_path)

    out = store.write_json_payload(
        request_id="req-abc",
        attempt_id="attempt-1",
        payload={"ok": True, "items": [3, 1, 2]},
    )

    assert out["path"].endswith("/re/req-abc/attempt-1.json")
    assert out["checksum"].startswith("sha256:")
    assert out["size_bytes"] > 0
    assert store.read_json_payload(path=out["path"], expected_checksum=out["checksum"]) == {
        "items": [3, 1, 2],
        "ok": True,
    }


def test_task_payload_store_supports_pydantic_payload(tmp_path) -> None:
    store = TaskPayloadStore(tmp_path)

    out = store.write_json_payload(
        request_id="req-model",
        attempt_id="attempt-1",
        payload=_PayloadModel(request_id="req-model", values=[1, 2]),
    )

    assert store.read_json_payload(path=out["path"], expected_checksum=out["checksum"]) == {
        "request_id": "req-model",
        "values": [1, 2],
    }


def test_task_payload_store_rejects_unsafe_components(tmp_path) -> None:
    store = TaskPayloadStore(tmp_path)

    with pytest.raises(TaskPayloadStoreError):
        store.write_json_payload(request_id="../escape", attempt_id="attempt-1", payload={})
    with pytest.raises(TaskPayloadStoreError):
        store.write_json_payload(request_id="req", attempt_id="../escape", payload={})


def test_task_payload_store_rejects_checksum_mismatch(tmp_path) -> None:
    store = TaskPayloadStore(tmp_path)
    out = store.write_json_payload(request_id="req", attempt_id="attempt-1", payload={"x": 1})

    with pytest.raises(TaskPayloadStoreError, match="checksum mismatch"):
        store.read_json_payload(path=out["path"], expected_checksum="sha256:bad")


def test_task_payload_store_rejects_read_outside_root(tmp_path) -> None:
    store = TaskPayloadStore(tmp_path / "root")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(TaskPayloadStoreError, match="outside root"):
        store.read_json_payload(path=outside)
