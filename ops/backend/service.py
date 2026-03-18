from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import datetime as dt
import json
import threading
import time
from typing import Any

import httpx

from .config import OpsBackendConfig


class DeployServiceError(RuntimeError):
    """Raised when the ops backend cannot fulfill a request."""


class DeployService(ABC):
    @abstractmethod
    def get_deploy_state(self, *, actor_type: str | None = None, model_query: str | None = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def recycle_actor(self, *, actor_type: str, model_name: str | None = None, actor_name: str | None = None) -> dict:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class DirectMintOpsService(DeployService):
    def __init__(self, config: OpsBackendConfig):
        self.config = config
        self._ray_lock = threading.Lock()
        self._ray_module: Any | None = None

    def _ensure_config(self) -> None:
        try:
            self.config.validate_runtime()
        except ValueError as exc:
            raise DeployServiceError(str(exc)) from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any] | list[Any] | str:
        self._ensure_config()
        url = f"{self.config.mint_base_url}{path}"
        try:
            with httpx.Client(timeout=timeout_s or self.config.timeout_s) as client:
                response = client.request(method, url, headers=self._headers(), json=payload)
        except httpx.HTTPError as exc:
            raise DeployServiceError(f"mint request failed: {method} {path}: {exc}") from exc

        if response.status_code >= 400:
            body_preview = response.text.strip()[:400]
            raise DeployServiceError(f"mint request failed: {method} {path} -> {response.status_code}: {body_preview}")

        if not response.text:
            return ""
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise DeployServiceError(f"mint returned invalid json for {method} {path}") from exc

    def _get_ray(self) -> Any:
        self._ensure_config()
        with self._ray_lock:
            if self._ray_module is not None:
                return self._ray_module
            import ray

            if ray.is_initialized():
                self._ray_module = ray
                return ray
            ray.init(address=self.config.ray_address, ignore_reinit_error=True, logging_level="ERROR")
            self._ray_module = ray
            return ray

    def _collect_nodes(self, ray: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
        nodes: list[dict[str, Any]] = []
        id_to_ip: dict[str, str] = {}
        for node in ray.nodes():
            node_id = str(node.get("NodeID", ""))
            ip = str(node.get("NodeManagerAddress", ""))
            id_to_ip[node_id] = ip
            resources = node.get("Resources", {}) or {}
            nodes.append(
                {
                    "node_id": node_id,
                    "node_id_short": node_id[:8],
                    "ip": ip,
                    "alive": bool(node.get("Alive", False)),
                    "gpu_total": int(float(resources.get("GPU", 0))),
                    "cpu_total": int(float(resources.get("CPU", 0))),
                }
            )
        nodes.sort(key=lambda item: (not item["alive"], -item["gpu_total"], item["ip"]))
        return nodes, id_to_ip

    def _collect_placement_groups(self, ray: Any, *, id_to_ip: dict[str, str]) -> list[dict[str, Any]]:
        raw = ray.util.placement_group_table()
        placement_groups = list(raw.values()) if isinstance(raw, dict) else list(raw)
        out: list[dict[str, Any]] = []
        for pg in placement_groups:
            state = str(pg.get("state", "UNKNOWN"))
            if state == "REMOVED" and not self.config.include_removed_pg:
                continue
            bundles = pg.get("bundles") or {}
            bundles_to_node_id = pg.get("bundles_to_node_id") or {}
            node_bundle_counts: Counter[str] = Counter()
            node_gpu_counts: defaultdict[str, float] = defaultdict(float)
            requested_gpu = 0.0
            for bundle_idx, resources in bundles.items():
                gpu = float((resources or {}).get("GPU", 0) or 0)
                requested_gpu += gpu
                node_id = bundles_to_node_id.get(bundle_idx) or bundles_to_node_id.get(str(bundle_idx)) or ""
                if not node_id:
                    continue
                ip = id_to_ip.get(node_id, f"<{str(node_id)[:8]}>")
                node_bundle_counts[ip] += 1
                node_gpu_counts[ip] += gpu
            out.append(
                {
                    "pg_id": str(pg.get("placement_group_id", "")),
                    "name": str(pg.get("name", "")),
                    "state": state,
                    "strategy": str(pg.get("strategy", "")),
                    "bundle_count": len(bundles),
                    "requested_gpu": requested_gpu,
                    "node_distribution": {
                        ip: {"bundles": node_bundle_counts[ip], "gpu": node_gpu_counts[ip]}
                        for ip in sorted(node_bundle_counts.keys())
                    },
                }
            )
        state_order = {"PENDING": 0, "CREATED": 1}
        out.sort(key=lambda item: (state_order.get(item["state"], 2), item["name"]))
        return out

    def _collect_ray_actor_details(self, ray: Any, *, id_to_ip: dict[str, str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            from ray.util.state import list_actors

            gcs_addr = str(ray.get_runtime_context().gcs_address)
            head_ip = gcs_addr.split(":", 1)[0]
            dashboard_url = f"http://{head_ip}:8265"
            actors = list_actors(address=dashboard_url, limit=10000)
            for actor in actors:
                data = actor.asdict() if hasattr(actor, "asdict") else actor.__dict__
                req = actor.required_resources or {}
                node_id = actor.node_id or ""
                rows.append(
                    {
                        "name": actor.name or "",
                        "class_name": actor.class_name or "",
                        "state": actor.state or "",
                        "namespace": actor.ray_namespace or "",
                        "node_id": node_id,
                        "ip": id_to_ip.get(node_id, ""),
                        "pid": actor.pid,
                        "num_gpus": float(req.get("GPU", 0)),
                        "start_time_ms": self._coerce_epoch_ms(
                            data.get("start_time_ms"),
                            data.get("start_time"),
                            data.get("StartTime"),
                            data.get("Timestamp"),
                        ),
                    }
                )
        except Exception:
            try:
                raw = ray.state.actors()  # type: ignore[attr-defined]
            except Exception as exc:
                raise DeployServiceError(f"ray actor query failed: {exc}") from exc
            for info in raw.values():
                req = info.get("RequiredResources") or {}
                node_id = info.get("Address", {}).get("NodeID", "")
                rows.append(
                    {
                        "name": info.get("Name", ""),
                        "class_name": info.get("ActorClassName", ""),
                        "state": info.get("State", ""),
                        "namespace": info.get("Namespace", ""),
                        "node_id": node_id,
                        "ip": id_to_ip.get(node_id, ""),
                        "pid": info.get("Pid", 0),
                        "num_gpus": float(req.get("GPU", 0)),
                        "start_time_ms": self._coerce_epoch_ms(
                            info.get("StartTime"),
                            info.get("Timestamp"),
                        ),
                    }
                )
        rows.sort(key=lambda item: (-item["num_gpus"], item["state"], item["class_name"], item["name"]))
        return rows

    @staticmethod
    def _coerce_epoch_ms(*values: Any) -> int | None:
        for value in values:
            if value in (None, "", 0, 0.0):
                continue
            try:
                millis = int(float(value))
            except (TypeError, ValueError):
                continue
            if millis > 0:
                return millis
        return None

    @staticmethod
    def _derive_managed_actor_status(*, actor_payload: dict[str, Any], pg_state: str | None, ray_states: list[str]) -> tuple[str, str]:
        actor_name = str(actor_payload.get("actor_name") or "").strip()
        creating = bool(actor_payload.get("creating"))
        normalized_ray_states = [str(x or "").strip().upper() for x in ray_states if str(x or "").strip()]
        alive = any(state == "ALIVE" for state in normalized_ray_states)
        pending = str(pg_state or "").strip().upper() == "PENDING"
        if pending:
            return "pending_pg", f"placement group pending for {actor_name or 'actor'}"
        if creating:
            return "creating", "resource pool marks actor as creating"
        if alive:
            return "ready", "ray actor alive"
        if normalized_ray_states:
            return "ray_not_alive", f"ray actor states={','.join(normalized_ray_states)}"
        return "unknown", "actor tracked by resource pool but no alive ray actor observed"

    @staticmethod
    def _build_rebuild_model_options(managed_actors: list[dict[str, Any]]) -> list[str]:
        models: list[str] = []
        seen: set[str] = set()
        try:
            from tinker_server.backend.model_registry import MODEL_CONFIGS

            for model in sorted(MODEL_CONFIGS.keys()):
                if model in seen:
                    continue
                seen.add(model)
                models.append(model)
        except Exception:
            pass

        for actor in managed_actors:
            model = str(actor.get("base_model") or "").strip()
            if not model or model in seen:
                continue
            seen.add(model)
            models.append(model)
        return models

    @staticmethod
    def _normalize_actor(actor: dict[str, Any]) -> dict[str, Any]:
        return {
            "actor_name": str(actor.get("actor_name") or ""),
            "actor_type": str(actor.get("actor_type") or ""),
            "base_model": str(actor.get("base_model") or ""),
            "num_gpus": int(actor.get("num_gpus") or 0),
            "idle_time": float(actor.get("idle_time") or 0.0),
            "protected": bool(actor.get("protected")),
            "current_session": actor.get("current_session"),
            "pg_name": str(actor.get("pg_name") or ""),
            "creating": bool(actor.get("creating")),
            "ops_pg_state": actor.get("ops_pg_state"),
            "ops_alive_node_ips": list(actor.get("ops_alive_node_ips") or []),
            "ops_ray_states": list(actor.get("ops_ray_states") or []),
            "ops_status": str(actor.get("ops_status") or "unknown"),
            "ops_status_reason": str(actor.get("ops_status_reason") or ""),
            "ops_started_at_utc": actor.get("ops_started_at_utc"),
            "ops_lifetime_seconds": actor.get("ops_lifetime_seconds"),
            "metadata": actor.get("metadata") or {},
        }

    def get_deploy_state(self, *, actor_type: str | None = None, model_query: str | None = None) -> dict:
        actors_payload = self._request_json("GET", "/api/v1/actors")
        if not isinstance(actors_payload, dict):
            raise DeployServiceError("mint /api/v1/actors returned unexpected payload")
        managed_actors = list(actors_payload.get("actors", []))

        ray = self._get_ray()
        nodes, id_to_ip = self._collect_nodes(ray)
        placement_groups = self._collect_placement_groups(ray, id_to_ip=id_to_ip)
        ray_actor_details = self._collect_ray_actor_details(ray, id_to_ip=id_to_ip)

        pg_name_to_state = {
            str(pg.get("name") or ""): str(pg.get("state") or "")
            for pg in placement_groups
            if str(pg.get("name") or "")
        }
        actor_name_to_ray_states: defaultdict[str, list[str]] = defaultdict(list)
        actor_name_to_alive_node_ips: defaultdict[str, list[str]] = defaultdict(list)
        actor_name_to_latest_start_ms: dict[str, int] = {}
        for actor in ray_actor_details:
            name = str(actor.get("name") or "")
            state = str(actor.get("state") or "").strip().upper()
            ip = str(actor.get("ip") or "").strip()
            start_time_ms = actor.get("start_time_ms")
            if name and state:
                actor_name_to_ray_states[name].append(state)
            if name and state == "ALIVE" and ip and ip not in actor_name_to_alive_node_ips[name]:
                actor_name_to_alive_node_ips[name].append(ip)
            if name and isinstance(start_time_ms, int):
                actor_name_to_latest_start_ms[name] = max(actor_name_to_latest_start_ms.get(name, 0), start_time_ms)

        normalized_actors = []
        now_ts = time.time()
        for actor in managed_actors:
            actor_name = str(actor.get("actor_name") or "").strip()
            pg_name = str(actor.get("pg_name") or "").strip()
            pg_state = pg_name_to_state.get(pg_name) if pg_name else None
            ray_states = actor_name_to_ray_states.get(actor_name, [])
            latest_start_ms = actor_name_to_latest_start_ms.get(actor_name)
            deploy_status, deploy_reason = self._derive_managed_actor_status(
                actor_payload=actor,
                pg_state=pg_state,
                ray_states=ray_states,
            )
            item = dict(actor)
            item["ops_pg_state"] = pg_state
            item["ops_alive_node_ips"] = list(actor_name_to_alive_node_ips.get(actor_name, []))
            item["ops_ray_states"] = list(ray_states)
            item["ops_status"] = deploy_status
            item["ops_status_reason"] = deploy_reason
            if latest_start_ms:
                item["ops_started_at_utc"] = dt.datetime.fromtimestamp(latest_start_ms / 1000, tz=dt.timezone.utc).isoformat()
                item["ops_lifetime_seconds"] = max(0.0, now_ts - latest_start_ms / 1000)
            else:
                item["ops_started_at_utc"] = None
                item["ops_lifetime_seconds"] = None
            normalized_actors.append(self._normalize_actor(item))

        if actor_type:
            normalized_actors = [actor for actor in normalized_actors if actor["actor_type"] == actor_type]
        if model_query:
            needle = model_query.strip().lower()
            normalized_actors = [
                actor for actor in normalized_actors if needle in actor["base_model"].lower() or needle in actor["actor_name"].lower()
            ]
        normalized_actors.sort(key=lambda item: (item["actor_type"], item["base_model"], item["actor_name"]))

        pending_pg_count = sum(1 for pg in placement_groups if str(pg.get("state")) == "PENDING")
        alive_ray_actor_count = sum(1 for actor in ray_actor_details if str(actor.get("state") or "").upper() == "ALIVE")
        rebuild_model_options = self._build_rebuild_model_options(managed_actors)

        return {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mint_base_url": self.config.mint_base_url,
            "ray_address": self.config.ray_address,
            "summary": {
                "gpu_total": int((ray.cluster_resources() or {}).get("GPU", 0)),
                "gpu_available": int((ray.available_resources() or {}).get("GPU", 0)),
                "actors": len(normalized_actors),
                "total_gpus_used": int(actors_payload.get("total_gpus_used") or 0),
                "pending_placement_groups": pending_pg_count,
                "nodes_alive": sum(1 for node in nodes if node["alive"]),
                "ray_actors_alive": alive_ray_actor_count,
            },
            "rebuild_model_options": rebuild_model_options,
            "actors": normalized_actors,
            "ray": {
                "nodes": nodes,
                "placement_groups": placement_groups,
                "actor_details": ray_actor_details,
            },
        }

    def recycle_actor(self, *, actor_type: str, model_name: str | None = None, actor_name: str | None = None) -> dict:
        payload: dict[str, Any] = {"actor_type": actor_type}
        if model_name:
            payload["model_name"] = model_name
        if actor_name:
            payload["actor_name"] = actor_name
        response = self._request_json(
            "POST",
            "/api/v1/actors/kill",
            payload=payload,
            timeout_s=max(self.config.timeout_s, 60.0),
        )
        if not isinstance(response, dict):
            raise DeployServiceError("mint /api/v1/actors/kill returned unexpected payload")
        return {
            **response,
            "action": "recycle_actor",
            "note": "recycle actor also cleans the actor placement group when backend supports actor_name_pg cleanup",
        }

    def _create_session(self, *, tag: str) -> str:
        response = self._request_json(
            "POST",
            "/api/v1/create_session",
            payload={
                "tags": [tag],
                "user_metadata": {},
                "sdk_version": "ops.backend",
            },
        )
        if not isinstance(response, dict) or not response.get("session_id"):
            raise DeployServiceError("create_session returned unexpected payload")
        return str(response["session_id"])

    def _create_sampling_session(self, *, session_id: str, model: str) -> str:
        response = self._request_json(
            "POST",
            "/api/v1/create_sampling_session",
            payload={
                "session_id": session_id,
                "sampling_session_seq_id": 0,
                "base_model": model,
            },
            timeout_s=max(self.config.timeout_s, 60.0),
        )
        if not isinstance(response, dict) or not response.get("sampling_session_id"):
            raise DeployServiceError("create_sampling_session returned unexpected payload")
        return str(response["sampling_session_id"])

    def _create_model(self, *, session_id: str, model: str, lora_rank: int) -> str:
        response = self._request_json(
            "POST",
            "/api/v1/create_model",
            payload={
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": model,
                "user_metadata": {},
                "lora_config": {"rank": int(lora_rank)},
            },
            timeout_s=max(self.config.timeout_s, 60.0),
        )
        if not isinstance(response, dict) or not response.get("request_id"):
            raise DeployServiceError("create_model returned unexpected payload")
        return str(response["request_id"])

    def _poll_future(self, *, request_id: str, timeout_s: float, interval_s: float) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        last_status: str | None = None
        while time.time() < deadline:
            response = self._request_json(
                "POST",
                "/api/v1/retrieve_future",
                payload={"request_id": request_id},
                timeout_s=self.config.timeout_s,
            )
            if isinstance(response, dict) and not response.get("error"):
                future_status = str(response.get("status") or response.get("future_status") or "").upper()
                last_status = future_status or last_status
                if response.get("ready") is True or future_status in {"READY", "SUCCESS", "COMPLETED"}:
                    return response
                if future_status in {"PENDING", "RUNNING", "CREATING", ""}:
                    time.sleep(interval_s)
                    continue
                if "result" in response and "error" not in response:
                    return response
                raise DeployServiceError(f"retrieve_future failed: {response}")
            time.sleep(interval_s)
        raise DeployServiceError(f"retrieve_future timeout for request_id={request_id}, last_status={last_status}")

    def _sample_ping(self, *, sampling_session_id: str, poll_timeout_s: float, poll_interval_s: float) -> dict[str, Any]:
        response = self._request_json(
            "POST",
            "/api/v1/asample",
            payload={
                "sampling_session_id": sampling_session_id,
                "num_samples": 1,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4]}]},
                "sampling_params": {"max_tokens": 1, "temperature": 0.0},
            },
        )
        if not isinstance(response, dict) or not response.get("request_id"):
            raise DeployServiceError("asample returned unexpected payload")
        return self._poll_future(
            request_id=str(response["request_id"]),
            timeout_s=poll_timeout_s,
            interval_s=poll_interval_s,
        )

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
        if kind not in {"vllm", "training"}:
            raise DeployServiceError("kind must be one of: vllm, training")
        if not models:
            raise DeployServiceError("at least one model is required")

        results: list[dict[str, Any]] = []
        for model in models:
            item: dict[str, Any] = {"model": model, "kind": kind, "status": "UNKNOWN"}
            t0 = time.time()
            try:
                session_id = self._create_session(tag="ops.backend:actor-rebuild")
                item["session_id"] = session_id
                if kind == "vllm":
                    sampling_session_id = self._create_sampling_session(session_id=session_id, model=model)
                    item["sampling_session_id"] = sampling_session_id
                    if sample_ping:
                        item["future"] = self._sample_ping(
                            sampling_session_id=sampling_session_id,
                            poll_timeout_s=poll_timeout_s,
                            poll_interval_s=poll_interval_s,
                        )
                else:
                    request_id = self._create_model(session_id=session_id, model=model, lora_rank=lora_rank)
                    item["future"] = self._poll_future(
                        request_id=request_id,
                        timeout_s=poll_timeout_s,
                        interval_s=poll_interval_s,
                    )
                item["status"] = "PASS"
            except Exception as exc:
                item["status"] = "FAIL"
                item["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                item["elapsed_s"] = round(time.time() - t0, 2)
            results.append(item)

        return {"ok": all(item.get("status") == "PASS" for item in results), "results": results}
