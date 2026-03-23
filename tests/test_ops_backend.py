from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ops.backend.config import OpsBackendConfig
from ops.backend.main import create_app
from ops.backend.service import DeployService


class StubDeployService(DeployService):
    def __init__(self) -> None:
        self.recycle_calls: list[tuple[str, str | None, str | None]] = []
        self.rebuild_calls: list[tuple[str, list[str]]] = []

    def get_deploy_state(self, *, actor_type: str | None = None, model_query: str | None = None) -> dict:
        return {
            "generated_at_utc": "2026-03-17T12:00:00Z",
            "mint_base_url": "http://127.0.0.1:18000",
            "ray_address": "192.168.0.10:6379",
            "summary": {
                "gpu_total": 32,
                "gpu_available": 8,
                "actors": 1,
                "total_gpus_used": 24,
                "pending_placement_groups": 2,
                "nodes_alive": 4,
                "ray_actors_alive": 18,
            },
            "rebuild_model_options": ["Qwen/Qwen3-30B-A3B-Instruct-2507"],
            "actors": [
                {
                    "actor_name": "tinker_vllm_qwen3",
                    "actor_type": actor_type or "vllm",
                    "base_model": model_query or "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "num_gpus": 8,
                    "idle_time": 4.0,
                    "protected": False,
                    "current_session": None,
                    "pg_name": "pg-qwen3",
                    "creating": False,
                    "ops_pg_state": "CREATED",
                    "ops_alive_node_ips": ["192.168.37.160"],
                    "ops_ray_states": ["ALIVE"],
                    "ops_status": "ready",
                    "ops_status_reason": "healthy",
                    "ops_started_at_utc": "2026-03-17T11:00:00Z",
                    "ops_lifetime_seconds": 3600.0,
                    "metadata": {},
                }
            ],
            "ray": {"nodes": [], "placement_groups": [], "actor_details": []},
        }

    def recycle_actor(self, *, actor_type: str, model_name: str | None = None, actor_name: str | None = None) -> dict:
        self.recycle_calls.append((actor_type, model_name, actor_name))
        return {"ok": True, "actor_type": actor_type, "model_name": model_name, "actor_name": actor_name}

    def rebuild_actor(
        self,
        *,
        kind: str,
        models: list[str],
        sample_ping: bool = False,
        lora_rank: int = 16,
        poll_timeout_s: float = 900.0,
        poll_interval_s: float = 2.0,
    ) -> dict:
        self.rebuild_calls.append((kind, models))
        return {"ok": True, "results": [{"kind": kind, "models": models}]}


def make_client() -> tuple[TestClient, StubDeployService]:
    repo_root = Path(__file__).resolve().parents[1]
    config = OpsBackendConfig.from_repo_root(repo_root)
    config.mint_base_url = "http://127.0.0.1:18000"
    config.ray_address = "192.168.0.10:6379"
    service = StubDeployService()
    app = create_app(config=config, service=service)
    return TestClient(app), service


def test_health_route_reports_explicit_config() -> None:
    client, _service = make_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["config_ready"] is True
    assert payload["mint_base_url"] == "http://127.0.0.1:18000"
    assert payload["ray_address"] == "192.168.0.10:6379"


def test_deploy_state_route_returns_payload() -> None:
    client, _service = make_client()
    response = client.get("/api/deploy/state", params={"actor_type": "vllm", "model_query": "Qwen"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["actors"] == 1
    assert payload["actors"][0]["actor_type"] == "vllm"
    assert payload["actors"][0]["base_model"] == "Qwen"


def test_recycle_route_proxies_payload() -> None:
    client, service = make_client()
    response = client.post(
        "/api/deploy/actors/recycle",
        json={
            "actor_type": "vllm",
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "actor_name": "tinker_vllm_qwen3",
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert service.recycle_calls == [("vllm", "Qwen/Qwen3-30B-A3B-Instruct-2507", "tinker_vllm_qwen3")]


def test_rebuild_route_proxies_models() -> None:
    client, service = make_client()
    response = client.post(
        "/api/deploy/actors/rebuild",
        json={
            "kind": "training",
            "models": ["Qwen/Qwen3-235B-A22B-Instruct-2507"],
            "sample_ping": False,
            "lora_rank": 32,
            "poll_timeout_s": 1200,
            "poll_interval_s": 3,
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert service.rebuild_calls == [("training", ["Qwen/Qwen3-235B-A22B-Instruct-2507"])]
