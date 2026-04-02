from __future__ import annotations

from tinker_server.config import ServerConfig


def test_issue_439_actor_env_aliases_prefer_mint_api_work_queue_name() -> None:
    cfg = ServerConfig.from_sources(
        environ={
            "MINT_API_WORK_QUEUE_ACTOR_NAME": "mint-queue",
            "TINKER_API_WORK_QUEUE_ACTOR_NAME": "legacy-queue",
        },
        config_path=None,
        config_file=None,
    )

    assert cfg.api_work_queue_actor_name == "mint-queue"


def test_issue_439_actor_env_aliases_fallback_to_legacy_names() -> None:
    cfg = ServerConfig.from_sources(
        environ={
            "TINKER_API_WORK_QUEUE_ACTOR_NAME": "legacy-queue",
            "TINKER_CAPACITY_MANAGER_ACTOR_NAME": "legacy-capacity",
        },
        config_path=None,
        config_file=None,
    )

    assert cfg.api_work_queue_actor_name == "legacy-queue"
    assert cfg.capacity_manager_actor_name == "legacy-capacity"
