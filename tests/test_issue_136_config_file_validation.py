import pytest

from tinker_server.config import ServerConfig
from tinker_server.config_file import load_tinker_config_file


def test_config_file_load_ok(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[server]",
                "max_loras = 13",
                "max_cpu_loras = 19",
                "max_lora_rank = 23",
                "vllm_attention_backend = 'FLASH_ATTN'",
                "",
                "[sampling]",
                "max_inflight_sample_tasks = 7",
                "max_concurrent_samples_per_request = 3",
                "",
                "[paths]",
                'pfs_runtime_env_root = "/vePFS/runtime/tinker-py31213"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_tinker_config_file(p)
    assert cfg.server.max_loras == 13
    assert cfg.server.vllm_attention_backend == "FLASH_ATTN"
    assert cfg.sampling.max_inflight_sample_tasks == 7
    assert cfg.paths.pfs_runtime_env_root == "/vePFS/runtime/tinker-py31213"


def test_server_config_vllm_attention_backend_prefers_env_over_file(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text("[server]\nvllm_attention_backend = 'TRITON_ATTN'\n", encoding="utf-8")
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={"TINKER_VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        config_path=None,
        config_file=file_cfg,
    )

    assert cfg.vllm_attention_backend == "FLASH_ATTN"


def test_config_file_unknown_key_fails_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("unknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_tinker_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_type_mismatch_fails_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[server]\nmax_loras = 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_tinker_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_legacy_runtime_path_keys_fail_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(
        "\n".join(
            [
                "[paths]",
                'pfs_runtime_env_root = "/vePFS/runtime/tinker-py31213"',
                'pfs_verl_path = "/vePFS/verl"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_tinker_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_sampling_window_loads(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[sampling]",
                "sample_coalesce = true",
                "sample_coalesce_window_ms = 12.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_tinker_config_file(p)
    assert cfg.sampling.sample_coalesce is True
    assert cfg.sampling.sample_coalesce_window_ms == 12.5


def test_config_file_future_replay_settings_load(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[future_store]",
                "replay_root_dir = '/tmp/future-replay'",
                "replay_hot_ttl_s = 30",
                "replay_disk_ttl_s = 300",
                "replay_sweep_interval_s = 600",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_tinker_config_file(p)
    assert cfg.future_store.replay_root_dir == "/tmp/future-replay"
    assert cfg.future_store.replay_hot_ttl_s == 30
    assert cfg.future_store.replay_disk_ttl_s == 300
    assert cfg.future_store.replay_sweep_interval_s == 600


def test_server_config_future_replay_root_defaults_to_dev_without_auth():
    cfg = ServerConfig.from_sources(environ={}, config_path=None, config_file=None)
    assert cfg.future_replay_root_dir == "/vePFS-Mindverse/share/mint-prod-dev/future-replay"


def test_server_config_future_replay_root_defaults_to_prod_with_auth():
    cfg = ServerConfig.from_sources(
        environ={"TINKER_API_KEY": "secret"},
        config_path=None,
        config_file=None,
    )
    assert cfg.future_replay_root_dir == "/vePFS-Mindverse/share/mint-prod-data/future-replay"


def test_server_config_future_replay_env_overrides_file_independently(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[future_store]",
                "replay_root_dir = '/tmp/from-file'",
                "replay_hot_ttl_s = 30",
                "replay_disk_ttl_s = 300",
                "replay_sweep_interval_s = 600",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={
            "MINT_FUTURE_REPLAY_ROOT_DIR": "/tmp/from-env",
            "MINT_FUTURE_REPLAY_HOT_TTL_S": "31",
            "MINT_FUTURE_REPLAY_DISK_TTL_S": "301",
            "MINT_FUTURE_REPLAY_SWEEP_INTERVAL_S": "601",
        },
        config_path=None,
        config_file=file_cfg,
    )

    assert cfg.future_replay_root_dir == "/tmp/from-env"
    assert cfg.future_replay_hot_ttl_s == 31.0
    assert cfg.future_replay_disk_ttl_s == 301.0
    assert cfg.future_replay_sweep_interval_s == 601.0


def test_server_config_future_replay_partial_env_override_preserves_untouched_file_values(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[future_store]",
                "replay_root_dir = '/tmp/from-file'",
                "replay_hot_ttl_s = 30",
                "replay_disk_ttl_s = 300",
                "replay_sweep_interval_s = 600",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={
            "MINT_FUTURE_REPLAY_DISK_TTL_S": "301",
        },
        config_path=None,
        config_file=file_cfg,
    )

    assert cfg.future_replay_root_dir == "/tmp/from-file"
    assert cfg.future_replay_hot_ttl_s == 30.0
    assert cfg.future_replay_disk_ttl_s == 301.0
    assert cfg.future_replay_sweep_interval_s == 600.0


def test_server_config_reads_usage_log_dir_from_env():
    cfg = ServerConfig.from_sources(
        environ={"TINKER_USAGE_LOG_DIR": "/vePFS/shared/billing"},
        config_path=None,
        config_file=None,
    )

    assert cfg.usage_log_dir == "/vePFS/shared/billing"


def test_server_config_reads_usage_log_dir_from_file(tmp_path):
    p = tmp_path / "usage.toml"
    p.write_text("[server]\nusage_log_dir = '/vePFS/shared/from-file'\n", encoding="utf-8")
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={},
        config_path=str(p),
        config_file=file_cfg,
    )

    assert cfg.usage_log_dir == "/vePFS/shared/from-file"


def test_server_config_fails_fast_for_non_postgres_usage_backend():
    with pytest.raises(ValueError, match="Unsupported usage backend 'sqlite'"):
        ServerConfig.from_sources(
            environ={"TINKER_USAGE_BACKEND": "sqlite"},
            config_path=None,
            config_file=None,
        )
