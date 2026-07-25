from __future__ import annotations

from pathlib import Path

from mint_server.backend.openpi.openpi_pi05_action_worker import (
    configure_jax_compilation_cache,
)


class _FakeConfig:
    def __init__(self) -> None:
        self.values = {
            "jax_persistent_cache_min_compile_time_secs": 0.0,
            "jax_persistent_cache_min_entry_size_bytes": -1,
        }
        self.updates: list[tuple[str, str]] = []

    def update(self, key: str, value: str) -> None:
        self.updates.append((key, value))


class _FakeJax:
    def __init__(self) -> None:
        self.config = _FakeConfig()


def test_configure_jax_compilation_cache_is_disabled_without_path(monkeypatch) -> None:
    monkeypatch.delenv("MINT_OPENPI_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    fake = _FakeJax()

    assert configure_jax_compilation_cache(fake) is None
    assert fake.config.updates == []


def test_configure_jax_compilation_cache_creates_mint_path(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "nested" / "jax-cache"
    monkeypatch.setenv("MINT_OPENPI_JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "lower-priority"))
    fake = _FakeJax()

    result = configure_jax_compilation_cache(fake)

    assert result == cache_dir.resolve()
    assert cache_dir.is_dir()
    assert fake.config.updates == [
        ("jax_compilation_cache_dir", str(cache_dir.resolve()))
    ]


def test_configure_jax_compilation_cache_accepts_standard_jax_env(
    monkeypatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "standard-jax-cache"
    monkeypatch.delenv("MINT_OPENPI_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    fake = _FakeJax()

    assert configure_jax_compilation_cache(fake) == cache_dir.resolve()
    assert fake.config.updates == [
        ("jax_compilation_cache_dir", str(cache_dir.resolve()))
    ]
