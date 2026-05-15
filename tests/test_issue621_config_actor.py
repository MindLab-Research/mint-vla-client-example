from __future__ import annotations

from dataclasses import replace

import pytest

from tinker_server.config import ServerConfig
from tinker_server.runtime_config import (
    CONFIG_ACTOR_DEFAULT_NAME,
    CONFIG_CLASS_ACTOR_CREATION_INPUT,
    CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV,
    CONFIG_CLASS_OBSERVABILITY,
    CONFIG_CLASS_SNAPSHOT_CONFIG,
    CONFIG_CLASS_TASK_STATE,
    REDACTED_VALUE,
    ConfigSnapshot,
    build_config_snapshot,
    classify_env,
    classify_env_key,
    config_actor_name,
    snapshot_fingerprint,
)


def test_config_actor_uses_stable_namespace_local_name_by_default() -> None:
    assert config_actor_name({}) == CONFIG_ACTOR_DEFAULT_NAME
    assert config_actor_name({"MINT_CONFIG_ACTOR_NAME": "custom_config"}) == "custom_config"
    assert config_actor_name({"MINT_CONFIG_ACTOR_NAME": "   "}) == CONFIG_ACTOR_DEFAULT_NAME


def test_runtime_config_classifies_bootstrap_actor_creation_snapshot_and_observability() -> None:
    assert classify_env_key("PFS_RUNTIME_ENV_ROOT") == CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV
    assert classify_env_key("MINT_RAY_JOB_WORKING_DIR") == CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV
    assert classify_env_key("MINT_MODEL_PLACEMENT_JSON") == CONFIG_CLASS_ACTOR_CREATION_INPUT
    assert classify_env_key("MINT_FUTURE_STORE_ACTOR_NAME") == CONFIG_CLASS_ACTOR_CREATION_INPUT
    assert classify_env_key("MINT_VLLM_MAX_NUM_SEQS") == CONFIG_CLASS_SNAPSHOT_CONFIG
    assert classify_env_key("MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S") == CONFIG_CLASS_SNAPSHOT_CONFIG
    assert classify_env_key("OTEL_EXPORTER_OTLP_ENDPOINT") == CONFIG_CLASS_OBSERVABILITY
    assert classify_env_key("MINT_TASK_STATE_STORE_DB_PATH") == CONFIG_CLASS_TASK_STATE

    grouped = classify_env(
        {
            "PFS_RUNTIME_ENV_ROOT": "/runtime",
            "MINT_FUTURE_STORE_ACTOR_NAME": "future",
            "MINT_VLLM_MAX_NUM_SEQS": "64",
            "MINT_VLLM_MAX_NUM_BATCHED_TOKENS": "4096",
            "OTEL_SERVICE_NAME": "mint",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=secret",
            "MINT_TASK_STATE_STORE_DB_PATH": "/tmp/task.sqlite3",
            "UNRELATED_ENV": "ignored",
        }
    )

    assert grouped[CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV]["PFS_RUNTIME_ENV_ROOT"] == "/runtime"
    assert grouped[CONFIG_CLASS_ACTOR_CREATION_INPUT]["MINT_FUTURE_STORE_ACTOR_NAME"] == "future"
    assert grouped[CONFIG_CLASS_SNAPSHOT_CONFIG]["MINT_VLLM_MAX_NUM_SEQS"] == "64"
    assert grouped[CONFIG_CLASS_SNAPSHOT_CONFIG]["MINT_VLLM_MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert grouped[CONFIG_CLASS_OBSERVABILITY]["OTEL_SERVICE_NAME"] == "mint"
    assert grouped[CONFIG_CLASS_OBSERVABILITY]["OTEL_EXPORTER_OTLP_HEADERS"] == REDACTED_VALUE
    assert grouped[CONFIG_CLASS_TASK_STATE]["MINT_TASK_STATE_STORE_DB_PATH"] == "/tmp/task.sqlite3"
    assert "UNRELATED_ENV" not in grouped["unclassified"]


def test_config_snapshot_is_read_only_shape_and_fingerprint_ignores_created_at() -> None:
    cfg = ServerConfig(
        future_store_actor_name="future-test",
        api_work_queue_actor_name="queue-test",
        config_path="/etc/mint/config.toml",
        api_key="secret-key",
        usage_pg_dsn="postgres://mint:secret@db/mint",
    )
    environ = {
        "PFS_RUNTIME_ENV_ROOT": "/runtime",
        "MINT_CONFIG_ACTOR_NAME": "mint_config",
        "MINT_VLLM_MAX_NUM_BATCHED_TOKENS": "4096",
    }

    first = build_config_snapshot(
        environ=environ,
        ray_namespace="mint-test",
        config=cfg,
        created_at=1.0,
    )
    second = build_config_snapshot(
        environ=environ,
        ray_namespace="mint-test",
        config=cfg,
        created_at=2.0,
    )

    assert isinstance(first, ConfigSnapshot)
    assert first.actor_name == "mint_config"
    assert first.ray_namespace == "mint-test"
    assert first.config_path == "/etc/mint/config.toml"
    assert first.server_config["future_store_actor_name"] == "future-test"
    assert first.server_config["api_key"] == REDACTED_VALUE
    assert first.server_config["usage_pg_dsn"] == REDACTED_VALUE
    assert first.env[CONFIG_CLASS_SNAPSHOT_CONFIG]["MINT_VLLM_MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert first.fingerprint == second.fingerprint
    assert snapshot_fingerprint(first.to_dict()) == first.fingerprint


def test_config_actor_exposes_no_mutating_api() -> None:
    from tinker_server.backend import config_actor

    assert hasattr(config_actor, "ensure_started")
    assert hasattr(config_actor, "get_snapshot")
    assert not hasattr(config_actor, "put")
    assert not hasattr(config_actor, "set_many")
    assert not hasattr(config_actor, "replace_snapshot")


def test_config_actor_detects_existing_snapshot_mismatch(monkeypatch) -> None:
    from tinker_server.backend import config_actor

    expected = build_config_snapshot(
        environ={"MINT_CONFIG_ACTOR_NAME": "mint_config"},
        ray_namespace="ns",
        config=ServerConfig(),
        created_at=1.0,
    )
    stale = replace(expected, fingerprint="stale")

    class FakeRef:
        pass

    class FakeRemoteMethod:
        def remote(self):
            return FakeRef()

    class FakeActor:
        get_snapshot = FakeRemoteMethod()

    class FakeRay:
        @staticmethod
        def get(_ref, timeout=None):
            return stale.to_dict()

    monkeypatch.setattr(config_actor, "ray", FakeRay)

    with pytest.raises(config_actor.ConfigActorSnapshotMismatchError, match="fingerprint mismatch"):
        config_actor._ensure_fingerprint_matches(FakeActor(), expected, timeout_s=1.0)


def test_config_actor_options_are_detached_namespace_local(monkeypatch) -> None:
    from tinker_server.backend import config_actor

    monkeypatch.setattr(config_actor, "RAY_NAMESPACE", "mint-ns")
    monkeypatch.setattr(config_actor, "PFS_PYTHONPATH", "/pythonpath")
    monkeypatch.setattr(
        config_actor,
        "actor_runtime_env",
        lambda *, pythonpath: {"env_vars": {"PYTHONPATH": pythonpath}},
    )
    monkeypatch.setattr(config_actor, "apply_detached_actor_resources", lambda options, ray_module: None)

    options = config_actor._actor_options(actor_name="mint_config")

    assert options["name"] == "mint_config"
    assert options["namespace"] == "mint-ns"
    assert options["lifetime"] == "detached"
    assert options["get_if_exists"] is True
    assert options["runtime_env"] == {"env_vars": {"PYTHONPATH": "/pythonpath"}}
