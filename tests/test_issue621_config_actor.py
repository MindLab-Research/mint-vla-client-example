from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mint_server.config import ServerConfig
from mint_server.runtime_config import (
    CONFIG_ACTOR_DEFAULT_NAME,
    CONFIG_CLASS_ACTOR_CREATION_INPUT,
    CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV,
    CONFIG_CLASS_OBSERVABILITY,
    CONFIG_CLASS_SNAPSHOT_CONFIG,
    CONFIG_CLASS_TASK_STATE,
    CONFIG_CLASS_UNCLASSIFIED,
    REDACTED_VALUE,
    ConfigSnapshot,
    actor_env_from_environ,
    actor_env_with_legacy_bridges,
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
    assert classify_env_key("MINT_MODEL_PLACEMENT_JSON") == CONFIG_CLASS_UNCLASSIFIED
    assert classify_env_key("MINT_MODEL_ACTOR_REPLICA_ID") == CONFIG_CLASS_UNCLASSIFIED
    assert classify_env_key("MINT_VLLM_MAX_NUM_SEQS") == CONFIG_CLASS_SNAPSHOT_CONFIG
    assert classify_env_key("MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S") == CONFIG_CLASS_SNAPSHOT_CONFIG
    assert classify_env_key("MINT_TOPOLOGY_CONFIG_PATH") == CONFIG_CLASS_SNAPSHOT_CONFIG
    assert classify_env_key("MINT_TOPOLOGY_STATE_PATH") == CONFIG_CLASS_SNAPSHOT_CONFIG
    assert classify_env_key("OTEL_EXPORTER_OTLP_ENDPOINT") == CONFIG_CLASS_OBSERVABILITY
    assert classify_env_key("MINT_DEPLOYMENT_ENV") == CONFIG_CLASS_OBSERVABILITY
    assert classify_env_key("MINT_CLUSTER_ID") == CONFIG_CLASS_OBSERVABILITY
    assert classify_env_key("MINT_TASK_STATE_STORE_DB_PATH") == CONFIG_CLASS_TASK_STATE
    assert classify_env_key("MINT_TASK_STATE_STORE_OWNER_TTL_S") == CONFIG_CLASS_TASK_STATE
    assert classify_env_key("VOLCENGINE_ACCESS_KEY") == CONFIG_CLASS_SNAPSHOT_CONFIG
    assert classify_env_key("VOLCENGINE_SECRET_KEY") == CONFIG_CLASS_SNAPSHOT_CONFIG

    grouped = classify_env(
        {
            "PFS_RUNTIME_ENV_ROOT": "/runtime",
            "MINT_VLLM_MAX_NUM_SEQS": "64",
            "MINT_VLLM_MAX_NUM_BATCHED_TOKENS": "4096",
            "OTEL_SERVICE_NAME": "mint",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=secret",
            "MINT_DEPLOYMENT_ENV": "prod",
            "MINT_CLUSTER_ID": "volcano",
            "MINT_TASK_STATE_STORE_DB_PATH": "/tmp/task.sqlite3",
            "MINT_TASK_STATE_STORE_OWNER_RENEW_S": "10",
            "VOLCENGINE_ACCESS_KEY": "volc-ak",
            "VOLCENGINE_SECRET_KEY": "volc-sk",
            "UNRELATED_ENV": "ignored",
        }
    )

    assert grouped[CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV]["PFS_RUNTIME_ENV_ROOT"] == "/runtime"
    assert grouped[CONFIG_CLASS_SNAPSHOT_CONFIG]["MINT_VLLM_MAX_NUM_SEQS"] == "64"
    assert grouped[CONFIG_CLASS_SNAPSHOT_CONFIG]["MINT_VLLM_MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert grouped[CONFIG_CLASS_OBSERVABILITY]["OTEL_SERVICE_NAME"] == "mint"
    assert grouped[CONFIG_CLASS_OBSERVABILITY]["OTEL_EXPORTER_OTLP_HEADERS"] == REDACTED_VALUE
    assert grouped[CONFIG_CLASS_OBSERVABILITY]["MINT_DEPLOYMENT_ENV"] == "prod"
    assert grouped[CONFIG_CLASS_OBSERVABILITY]["MINT_CLUSTER_ID"] == "volcano"
    assert grouped[CONFIG_CLASS_TASK_STATE]["MINT_TASK_STATE_STORE_DB_PATH"] == "/tmp/task.sqlite3"
    assert grouped[CONFIG_CLASS_TASK_STATE]["MINT_TASK_STATE_STORE_OWNER_RENEW_S"] == "10"
    assert grouped[CONFIG_CLASS_SNAPSHOT_CONFIG]["VOLCENGINE_ACCESS_KEY"] == "volc-ak"
    assert grouped[CONFIG_CLASS_SNAPSHOT_CONFIG]["VOLCENGINE_SECRET_KEY"] == REDACTED_VALUE
    assert "UNRELATED_ENV" not in grouped["unclassified"]


def test_config_snapshot_is_read_only_shape_and_fingerprint_ignores_created_at() -> None:
    cfg = ServerConfig(
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
    assert first.server_config["api_key"] == REDACTED_VALUE
    assert first.server_config["usage_pg_dsn"] == REDACTED_VALUE
    assert first.env[CONFIG_CLASS_SNAPSHOT_CONFIG]["MINT_VLLM_MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert first.actor_env["MINT_VLLM_MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert first.actor_env["MINT_CONFIG_ACTOR_NAME"] == "mint_config"
    assert first.fingerprint == second.fingerprint
    assert snapshot_fingerprint(first.to_dict()) == first.fingerprint


def test_actor_env_from_environ_keeps_real_values_for_actor_hydration() -> None:
    actor_env = actor_env_from_environ(
        {
            "PFS_RUNTIME_ENV_ROOT": "/runtime",
            "MINT_MODEL_PLACEMENT_JSON": "{}",
            "MINT_MODEL_ACTOR_REPLICA_ID": "replica-0",
            "MINT_VLLM_MAX_NUM_SEQS": "32",
            "MINT_TOPOLOGY_CONFIG_PATH": "/vePFS-Mindverse/share/mint/prod/runtime/topology.yaml",
            "MINT_TOPOLOGY_STATE_PATH": "/vePFS-Mindverse/share/mint/prod/runtime/topology_state.yaml",
            "MINT_DEPLOYMENT_ENV": "prod",
            "MINT_CLUSTER_ID": "volcano",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=secret",
            "MINT_TASK_STATE_STORE_DB_PATH": "/tmp/task.sqlite3",
            "MINT_TASK_STATE_STORE_OWNER_TTL_S": "30",
            "VOLCENGINE_ACCESS_KEY": "volc-ak",
            "VOLCENGINE_SECRET_KEY": "volc-sk",
            "VOLCENGINE_PROFILE": "prod",
            "MINT_API_KEY": "api-secret",
            "MINT_BASE_URL": "http://client-only",
            "MINT_NEW_FEATURE_FLAG": "1",
            "MINT_CONFIG_ACTOR_HYDRATE": "1",
            "UNRELATED": "ignored",
        }
    )

    assert "PFS_RUNTIME_ENV_ROOT" not in actor_env
    assert "MINT_MODEL_PLACEMENT_JSON" not in actor_env
    assert "MINT_MODEL_ACTOR_REPLICA_ID" not in actor_env
    assert actor_env["MINT_VLLM_MAX_NUM_SEQS"] == "32"
    assert actor_env["MINT_TOPOLOGY_CONFIG_PATH"] == "/vePFS-Mindverse/share/mint/prod/runtime/topology.yaml"
    assert actor_env["MINT_TOPOLOGY_STATE_PATH"] == "/vePFS-Mindverse/share/mint/prod/runtime/topology_state.yaml"
    assert actor_env["MINT_DEPLOYMENT_ENV"] == "prod"
    assert actor_env["MINT_CLUSTER_ID"] == "volcano"
    assert actor_env["OTEL_EXPORTER_OTLP_HEADERS"] == "Authorization=secret"
    assert actor_env["MINT_TASK_STATE_STORE_DB_PATH"] == "/tmp/task.sqlite3"
    assert actor_env["MINT_TASK_STATE_STORE_OWNER_TTL_S"] == "30"
    assert actor_env["VOLCENGINE_ACCESS_KEY"] == "volc-ak"
    assert actor_env["VOLCENGINE_SECRET_KEY"] == "volc-sk"
    assert actor_env["VOLCENGINE_PROFILE"] == "prod"
    assert "MINT_API_KEY" not in actor_env
    assert "MINT_BASE_URL" not in actor_env
    assert "MINT_NEW_FEATURE_FLAG" not in actor_env
    assert "MINT_CONFIG_ACTOR_HYDRATE" not in actor_env
    assert "UNRELATED" not in actor_env


def test_actor_env_with_legacy_bridges_lifts_volc_cli_credentials(tmp_path, monkeypatch) -> None:
    volc_home = tmp_path / ".volc"
    volc_home.mkdir()
    (volc_home / "credentials").write_text(
        "[prod]\n"
        "access_key_id = legacy-ak\n"
        "secret_access_key = legacy-sk\n"
        "session_token = legacy-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    actor_env = actor_env_with_legacy_bridges(
        {
            "VOLC_CLI_HOME": str(volc_home),
            "VOLC_PROFILE": "prod",
        }
    )

    assert actor_env["VOLCENGINE_ACCESS_KEY"] == "legacy-ak"
    assert actor_env["VOLCENGINE_SECRET_KEY"] == "legacy-sk"
    assert actor_env["VOLCENGINE_SESSION_TOKEN"] == "legacy-token"
    assert actor_env["VOLCENGINE_PROFILE"] == "prod"


def test_actor_env_with_legacy_bridges_prefers_modern_volcengine_env(tmp_path) -> None:
    volc_home = tmp_path / ".volc"
    volc_home.mkdir()
    (volc_home / "credentials").write_text(
        "[default]\naccess_key_id = legacy-ak\nsecret_access_key = legacy-sk\n",
        encoding="utf-8",
    )

    actor_env = actor_env_with_legacy_bridges(
        {
            "VOLC_CLI_HOME": str(volc_home),
            "VOLCENGINE_ACCESS_KEY": "modern-ak",
            "VOLCENGINE_SECRET_KEY": "modern-sk",
        }
    )

    assert actor_env["VOLCENGINE_ACCESS_KEY"] == "modern-ak"
    assert actor_env["VOLCENGINE_SECRET_KEY"] == "modern-sk"


def test_config_actor_exposes_no_mutating_api() -> None:
    from mint_server.backend import config_actor

    assert hasattr(config_actor, "ensure_started")
    assert hasattr(config_actor, "get_snapshot")
    assert not hasattr(config_actor, "put")
    assert not hasattr(config_actor, "set_many")
    assert not hasattr(config_actor, "replace_snapshot")


def test_config_actor_detects_existing_snapshot_mismatch(monkeypatch) -> None:
    from mint_server.backend import config_actor

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
    from mint_server.backend import config_actor

    monkeypatch.setattr(config_actor, "RAY_NAMESPACE", "mint-ns")
    monkeypatch.setattr(config_actor, "PFS_PYTHONPATH", "/pythonpath")
    monkeypatch.setattr(
        config_actor,
        "actor_runtime_env",
        lambda *, pythonpath, extra=None, include_config_snapshot=True: {
            "env_vars": {
                "PYTHONPATH": pythonpath,
                **(extra or {}),
                **({"MINT_CONFIG_ACTOR_HYDRATE": "1"} if include_config_snapshot else {}),
            }
        },
    )
    monkeypatch.setattr(config_actor, "apply_detached_actor_resources", lambda options, ray_module: None)

    options = config_actor._actor_options(actor_name="mint_config")

    assert options["name"] == "mint_config"
    assert options["namespace"] == "mint-ns"
    assert options["lifetime"] == "detached"
    assert options["get_if_exists"] is True
    assert options["runtime_env"] == {
        "env_vars": {
            "PYTHONPATH": "/pythonpath",
            "MINT_CONFIG_ACTOR_SELF": "1",
        }
    }


def test_actor_runtime_env_hydration_flag_is_default_and_not_extra_overridable(monkeypatch) -> None:
    from mint_server import config as server_config

    monkeypatch.setattr(server_config, "PFS_RUNTIME_ENV_ROOT", "/runtime")
    monkeypatch.setattr(server_config, "MINT_CODE_ROOT", "/repo")
    monkeypatch.setattr(server_config, "PFS_HF_MODULES_PATH", "/hf")
    monkeypatch.setattr(server_config, "RAY_NAMESPACE", "mint-test")
    monkeypatch.setenv("RAY_ADDRESS", "ray://127.0.0.1:10001")

    env_vars = server_config.actor_runtime_env_vars(
        pythonpath="/runtime/pythonpath",
        extra={"MINT_CONFIG_ACTOR_HYDRATE": "0"},
    )

    assert env_vars["MINT_CONFIG_ACTOR_HYDRATE"] == "1"


def test_only_config_actor_disables_config_actor_hydration() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    matches: list[tuple[str, str]] = []
    for path in sorted((repo_root / "mint_server").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "include_config_snapshot=False" in line:
                matches.append((str(path.relative_to(repo_root)), line.strip()))

    assert matches == [
        (
            "mint_server/backend/config_actor.py",
            "include_config_snapshot=False,",
        )
    ]


def test_config_hydration_applies_actor_env_once(monkeypatch) -> None:
    from mint_server import config_hydration

    class FakeRef:
        pass

    class FakeRemoteMethod:
        def remote(self):
            return FakeRef()

    class FakeActor:
        get_snapshot = FakeRemoteMethod()

    class FakeRay:
        @staticmethod
        def get_actor(actor_name, namespace=None):
            assert actor_name == "mint_config"
            assert namespace == "mint-ns"
            return FakeActor()

        @staticmethod
        def get(_ref, timeout=None):
            return {
                "actor_env": {
                    "MINT_VLLM_MAX_NUM_SEQS": "32",
                    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=secret",
                }
            }

    environ = {
        "MINT_CONFIG_ACTOR_HYDRATE": "1",
        "MINT_RAY_NAMESPACE": "mint-ns",
    }
    monkeypatch.setattr(config_hydration, "_HYDRATED", False)
    monkeypatch.setitem(__import__("sys").modules, "ray", FakeRay)

    assert config_hydration.hydrate_from_config_actor(environ) is True
    assert environ["MINT_VLLM_MAX_NUM_SEQS"] == "32"
    assert environ["OTEL_EXPORTER_OTLP_HEADERS"] == "Authorization=secret"

    environ["MINT_VLLM_MAX_NUM_SEQS"] = "64"
    assert config_hydration.hydrate_from_config_actor(environ) is True
    assert environ["MINT_VLLM_MAX_NUM_SEQS"] == "64"
