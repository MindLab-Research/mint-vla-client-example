import pytest

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
                "",
                "[sampling]",
                "max_inflight_sample_tasks = 7",
                "max_concurrent_samples_per_request = 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_tinker_config_file(p)
    assert cfg.server.max_loras == 13
    assert cfg.sampling.max_inflight_sample_tasks == 7


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
