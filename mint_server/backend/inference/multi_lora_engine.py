"""Multi-LoRA inference engine for multi-tenant serving.

Provides shared vLLM engine with per-sampling-session LoRA weights.
Each sampling session gets frozen weights via unique lora_int_id.
Supports GPU slots with CPU cache for overflow (LRU eviction).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from mint_server.backend.ray_cluster.async_ray_control import async_get_ray_ref
from mint_server.backend.core.model_registry import get_model_config
from mint_server.config import PFS_PYTHONPATH, RAY_NAMESPACE
from mint_server.config import config as server_config
from mint_server.logging_context import get_current_traceparent, run_async_with_otel_span
from mint_server.ray_utils import init_ray
from mint_server.runtime_env import join_pythonpath, sanitize_worker_pythonpath

import mint_server.backend.ray_cluster.ray_kill as ray_kill
from mint_server.backend.inference.lora_registry import LoRARegistry
from mint_server.backend.ray_cluster.ray_placement_groups import remove_named_placement_group
from mint_server.backend.actors.ray_keepalive import ray_get_with_model_actor_supervisor_keepalive
from mint_server.backend.actors.node_placement import (
    assert_node_ip_capacity,
    parse_model_gpu_placement,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default configuration for 80GB GPU
DEFAULT_MAX_LORAS = 64  # GPU slots (~2.5GB for rank-32 Qwen-7B)
DEFAULT_MAX_CPU_LORAS = 1024  # CPU cache for evicted adapters
DEFAULT_MAX_LORA_RANK = 64  # Maximum supported rank

# Env flag helper (keep local to avoid importing config parsing helpers).
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in {"0", "false", "no", "off"}


def _read_process_env_var(name: str) -> str:
    try:
        with open("/proc/self/environ", "rb") as f:
            raw = f.read().split(b"\0")
        prefix = f"{name}=".encode()
        for entry in raw:
            if entry.startswith(prefix):
                return entry[len(prefix) :].decode(errors="ignore")
    except Exception:
        pass
    return os.environ.get(name, "")


def _prepare_vllm_actor_runtime_env(env_vars: dict[str, str]) -> None:
    # Ray worker bootstrap must use node-local Ray state, not driver/API host
    # hints. Ray runtime_env overlays but does not reliably delete inherited
    # job env, so blank these keys instead of popping them. RAY_ADDRESS is not
    # emitted in runtime_env; worker entry wrappers delete inherited values
    # before Ray's Python bootstrap can consume them.
    env_vars.pop("RAY_ADDRESS", None)
    for key in (
        "MINT_RAY_TEMP_DIR",
        "MINT_RAY_NODE_IP_ADDRESS",
        "RAY_TMPDIR",
        "TMPDIR",
        "TMP",
        "TEMP",
        "RAY_CLIENT_ADDRESS",
        "MINT_RAY_CLIENT_ADDRESS",
    ):
        env_vars[key] = ""


# Fixed namespace for persistent actors (without this, each process gets random namespace)
PERSISTENT_NAMESPACE = RAY_NAMESPACE

_LORA_SESSION_INVALIDATORS: list[Callable[[str], int]] = []


def register_lora_session_invalidator(invalidator: Callable[[str], int]) -> None:
    if invalidator not in _LORA_SESSION_INVALIDATORS:
        _LORA_SESSION_INVALIDATORS.append(invalidator)


def _invalidate_model_session_loras(model_name: str) -> None:
    """Force multi-LoRA sessions for a model to reload after actor recreate."""
    try:
        invalidated = 0
        for invalidator in list(_LORA_SESSION_INVALIDATORS):
            invalidated += int(invalidator(model_name) or 0)
        if invalidated:
            logger.info(
                "Invalidated cached LoRA load state for %s session(s) on model=%s after vLLM actor recreate",
                invalidated,
                model_name,
            )
    except Exception as e:
        logger.warning(
            "Failed to invalidate cached LoRA load state for model=%s: %s: %s",
            model_name,
            type(e).__name__,
            e,
        )


def _get_actor_node_id(actor_handle: ray.actor.ActorHandle) -> str | None:
    """Get the node_id where an actor is running.

    Uses Ray's state API to get actor location.
    Returns None if unable to determine.
    """
    try:
        actor_id = actor_handle._actor_id
        # Convert ActorID to hex string for the state API
        actor_id_hex = actor_id.hex()
        from ray._private.state import actors as state_actors
        actor_info = state_actors(actor_id_hex)
        if actor_info:
            # NodeID is nested under Address
            address = actor_info.get("Address", {})
            return address.get("NodeID")
    except Exception as e:
        logger.debug(f"Could not get node_id for actor: {e}")
    return None


@dataclass
class GenerateResult:
    """Result of a generate call."""

    token_ids: list[int]
    logprobs: list[float] | None = None
    stop_reason: str | None = None
    routed_experts: list | None = None
    timing_total_s: float | None = None
    timing_first_tok_s: float | None = None


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None



class MultiLoRAInferenceEngine:
    """Shared inference engine with multi-tenant LoRA support.

    Single vLLM engine serves multiple sampling sessions, each with
    frozen LoRA weights identified by unique lora_int_id.

    Features:
    - GPU slots (max_loras) for hot adapters
    - CPU cache (max_cpu_loras) for evicted adapters
    - Automatic swap between GPU and CPU on demand
    - Per-session weight isolation (Tinker SDK semantics)
    """

    def __init__(
        self,
        model_path: str,
        model_name: str | None = None,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        max_num_seqs: int = 256,
        max_loras: int = DEFAULT_MAX_LORAS,
        max_cpu_loras: int = DEFAULT_MAX_CPU_LORAS,
        max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
        pinned_node_ip: str | None = None,
        actor_name: str | None = None,
        quantization: str | None = None,  # "fp8" for FP8 models like K2
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_lora_rank = max_lora_rank
        self.pinned_node_ip = pinned_node_ip.strip() if isinstance(pinned_node_ip, str) else None
        if not actor_name:
            raise ValueError("PersistentVLLMEngine requires a model-specific actor_name")
        self.actor_name = actor_name

        self.registry = LoRARegistry()
        self.server = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._path_load_lock = asyncio.Lock()
        self._path_load_futures: dict[str, asyncio.Future[None]] = {}

    async def initialize(self) -> None:
        """Initialize the shared vLLM engine with multi-LoRA support.

        Uses a detached Ray actor that survives server restarts.
        First tries to connect to existing actor, creates new one if not found.
        """
        async with self._init_lock:
            if self._initialized:
                return

            if not ray.is_initialized():
                # Use fixed namespace so detached actors can be found across process restarts
                init_ray(
                    namespace=PERSISTENT_NAMESPACE,
                    ignore_reinit_error=True,
                )

            # Try to get existing persistent actor
            # Note: ray.get_actor succeeds even for dead actors (name still registered)
            # We must verify the actor is alive AND engine is initialized
            try:
                self.server = ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                # Health check: try calling a method to verify actor is alive
                # This will raise RayActorError if actor is dead
                try:
                    try:
                        await async_get_ray_ref(self.server.__ray_ready__.remote(), timeout_s=5)
                        # Check if engine is actually initialized (not just actor alive)
                        # A broken actor can be "alive" but have failed engine init
                        engine_ready = await async_get_ray_ref(self.server.is_engine_ready.remote(), timeout_s=10)
                    except SystemExit as e:
                        if getattr(e, "code", None) == 15:
                            raise
                        raise RuntimeError(
                            f"ray.get(__ray_ready__/is_engine_ready) triggered SystemExit for {self.actor_name}: {e}"
                        ) from e
                    if not engine_ready:
                        logger.warning(
                            f"vLLM actor {self.actor_name} has broken engine, killing and recreating"
                        )
                        try:
                            ray_kill.kill(
                                self.server,
                                reason="vllm_engine_broken",
                                actor_name=self.actor_name,
                                namespace=PERSISTENT_NAMESPACE,
                                no_restart=True,
                            )
                        except Exception as kill_err:
                            logger.debug(f"Failed to kill broken actor: {kill_err}")
                        self.server = None
                        self._initialized = False
                    else:
                        logger.info(
                            f"Connected to existing persistent vLLM actor: {self.actor_name}"
                        )
                        self._initialized = True

                        # Publish existing actor through supervisor inventory.
                        # Include node_id for proper per-node GPU scheduling
                        from mint_server.backend.actors.model_actor_inventory import ActorType
                        from mint_server.backend.actors.model_actor_publication import (
                            BackendModelActorLaunch,
                            publish_backend_model_actor,
                        )
                        total_gpus = self.tensor_parallel_size * self.data_parallel_size
                        actor_node_id = _get_actor_node_id(self.server)
                        logger.info(f"Publishing existing actor {self.actor_name} through ModelActorSupervisor inventory (node={actor_node_id[:8] if actor_node_id else 'unknown'})")
                        publish_backend_model_actor(BackendModelActorLaunch(
                            actor_name=self.actor_name,
                            actor_type=ActorType.VLLM,
                            num_gpus=total_gpus,
                            actor_handle=self.server,
                            namespace=PERSISTENT_NAMESPACE,
                            base_model=self.model_path,
                            node_id=actor_node_id,
                        ))
                        return
                except ray.exceptions.RayActorError:
                    # Actor is dead, need to create new one
                    logger.warning(
                        f"vLLM actor {self.actor_name} is dead, creating new one"
                    )
                    # Must kill dead actor to free the name for reuse
                    # Ray keeps names registered even for dead actors
                    try:
                        ray_kill.kill(
                            self.server,
                            reason="vllm_actor_dead_free_name",
                            actor_name=self.actor_name,
                            namespace=PERSISTENT_NAMESPACE,
                            no_restart=True,
                        )
                        logger.debug(f"Killed dead actor to free name: {self.actor_name}")
                    except Exception as kill_err:
                        logger.debug(f"Could not kill dead actor: {kill_err}")
                    self.server = None
                    # Reset initialized flag so we'll create new actor below
                    self._initialized = False
                except ray.exceptions.GetTimeoutError:
                    # Actor might be busy (queued tasks) rather than dead.
                    # Killing on timeout will terminate active inference and training flows.
                    logger.warning(
                        f"vLLM actor {self.actor_name} __ray_ready__/is_engine_ready timed out; assuming busy and reusing actor"
                    )
                    logger.info(
                        f"Connected to existing persistent vLLM actor: {self.actor_name}"
                    )
                    self._initialized = True

                    from mint_server.backend.actors.model_actor_inventory import ActorType
                    from mint_server.backend.actors.model_actor_publication import (
                        BackendModelActorLaunch,
                        publish_backend_model_actor,
                    )
                    total_gpus = self.tensor_parallel_size * self.data_parallel_size
                    actor_node_id = _get_actor_node_id(self.server)
                    logger.info(
                        f"Publishing existing actor {self.actor_name} through ModelActorSupervisor inventory (node={actor_node_id[:8] if actor_node_id else 'unknown'})"
                    )
                    publish_backend_model_actor(BackendModelActorLaunch(
                        actor_name=self.actor_name,
                        actor_type=ActorType.VLLM,
                        num_gpus=total_gpus,
                        actor_handle=self.server,
                        namespace=PERSISTENT_NAMESPACE,
                        base_model=self.model_path,
                        node_id=actor_node_id,
                    ))
                    return
            except ValueError:
                # Actor doesn't exist, create new one
                logger.info(
                    f"No existing vLLM actor found, creating new detached actor: {self.actor_name}"
                )

            from verl.workers.config.model import HFModelConfig
            from verl.workers.config.rollout import CheckpointEngineConfig, RolloutConfig
            from verl.workers.rollout.replica import RolloutMode

            from mint_server.backend.training.verl.verl_inference import _create_extended_server_class

            # Pass multi-LoRA config to vLLM engine
            ExtendedVLLMHttpServer = _create_extended_server_class(
                max_loras=self.max_loras,
                max_cpu_loras=self.max_cpu_loras,
            )

            # Compute total GPUs needed for MoE models
            # For EP (expert parallelism), total_gpus = TP * DP
            total_gpus = self.tensor_parallel_size * self.data_parallel_size

            # Build engine_kwargs for expert parallelism
            # Pass enable_expert_parallel directly to vLLM, bypassing verl's worker-based EP
            # This allows vLLM to handle EP internally via multiprocessing
            engine_kwargs = {}
            if self.data_parallel_size > 1:
                engine_kwargs = {
                    "vllm": {
                        "enable_expert_parallel": True,
                    }
                }
                logger.info(f"Enabling expert parallelism via vLLM (DP={self.data_parallel_size})")

            # Determine effective context limit for vLLM
            model_cfg = get_model_config(self.model_path)
            # Use model registry's max_model_len (single source of truth)
            max_model_len = model_cfg.max_model_len
            # prompt_logprobs uses float32 log_softmax over [tokens, vocab], which can spike memory.
            # max_num_batched_tokens limits chunk size during prefill/logprobs to cap peak allocations.
            max_num_batched_tokens = model_cfg.max_num_batched_tokens
            if max_num_batched_tokens is None:
                max_num_batched_tokens = 4096 if max_model_len >= 32768 else 8192
            # verl calculates max_model_len = prompt_length + response_length
            # We split evenly, but this does NOT restrict actual prompt/response sizes:
            # - The split only affects verl's default max_new_tokens (response_length)
            # - Our generate() computes actual limit dynamically: max_model_len - len(prompt)
            # - A 32K model can do 25K prompt + 7K response, or 5K prompt + 27K response
            # The sum (total context window) is what matters, not the individual values.
            prompt_length = max_model_len // 2
            response_length = max_model_len - prompt_length
            logger.info(f"vLLM max_model_len={max_model_len} (prompt={prompt_length}, response={response_length})")

            # vLLM V1 engine can fail during multi-GPU init when cudagraph/compile is enabled
            # (worker subprocesses crash before reporting a useful exception). Eager mode trades
            # some throughput for deterministic startup.
            enforce_eager = bool(model_cfg.is_moe and total_gpus >= 4)

            # Configure rollout with multi-LoRA support
            # NOTE: Keep expert_parallel_size=1 to avoid verl's worker-based EP assertion
            # Expert parallelism is enabled via engine_kwargs instead
            enable_rollout_routing_replay = (server_config.router_replay_mode == "R3")
            rollout_config = RolloutConfig(
                name="vllm",
                tensor_model_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                prompt_length=prompt_length,
                response_length=response_length,
                max_model_len=max_model_len,
                max_num_seqs=self.max_num_seqs,
                dtype="auto",
                load_format="auto",
                enforce_eager=enforce_eager,
                enable_chunked_prefill=True,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_caching=True,
                disable_log_stats=True,
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
                data_parallel_size=1,
                expert_parallel_size=1,
                engine_kwargs=engine_kwargs,
                checkpoint_engine=CheckpointEngineConfig(backend="naive"),
                quantization=self.quantization,
                enable_rollout_routing_replay=enable_rollout_routing_replay,
            )
            if self.quantization:
                logger.info(f"vLLM quantization enabled: {self.quantization}")

            # Model config with multi-LoRA parameters
            model_config = HFModelConfig(
                path=self.model_path,
                trust_remote_code=True,
                lora_rank=self.max_lora_rank,
                lora_adapter_path=None,
            )

            logger.info(
                f"Initializing MultiLoRAInferenceEngine: "
                f"TP={self.tensor_parallel_size}, DP={self.data_parallel_size}, total_gpus={total_gpus}, "
                f"max_loras={self.max_loras}, max_cpu_loras={self.max_cpu_loras}, "
                f"max_lora_rank={self.max_lora_rank}"
            )

            # Create detached Ray actor with well-known name
            # lifetime="detached" ensures actor survives owner process termination
            # Request total_gpus for MoE expert parallelism
            # runtime_env prepends vLLM 0.12.0 from PFS for MoE LoRA support
            from mint_server.config import (
                actor_ld_library_path,
                actor_runtime_env_vars,
                otel_env_vars,
            )
            worker_pythonpath = join_pythonpath(
                "/vllm",
                sanitize_worker_pythonpath(
                    PFS_PYTHONPATH,
                    env_root=os.environ.get("PFS_RUNTIME_ENV_ROOT"),
                ),
            )
            env_vars = actor_runtime_env_vars(
                pythonpath=worker_pythonpath,
                extra={
                    "LD_LIBRARY_PATH": actor_ld_library_path(),
                    "MINT_VLLM_BASE_MODEL_NAME": str(self.model_name or self.model_path),
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "USE_TORCH": "1",
                    "USE_TF": "0",
                    "USE_FLAX": "0",
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    **otel_env_vars(),
                },
                include_ray_attach_hints=False,
            )
            env_vars.setdefault("VLLM_ATTENTION_BACKEND", server_config.vllm_attention_backend)
            if total_gpus >= 16:
                env_vars["MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE"] = "1"
            _prepare_vllm_actor_runtime_env(env_vars)

            actor_options: dict[str, object] = {
                "num_gpus": total_gpus,
                "name": self.actor_name,
                "namespace": PERSISTENT_NAMESPACE,
                "lifetime": "detached",
                "max_concurrency": int(os.environ.get("MINT_VLLM_ACTOR_MAX_CONCURRENCY", "64")),
                "runtime_env": {
                    "env_vars": {
                        **env_vars,
                    }
                },
            }
            if self.pinned_node_ip:
                actor_options["resources"] = {f"node:{self.pinned_node_ip}": 0.001}
                node_map = {
                    n.get("NodeManagerAddress"): n.get("NodeID")
                    for n in ray.nodes()
                    if n.get("Alive")
                }
                node_id = node_map.get(self.pinned_node_ip)
                if not node_id:
                    raise RuntimeError(
                        f"Requested pinned_node_ip={self.pinned_node_ip} for actor={self.actor_name} "
                        "but no alive Ray node has that IP"
                    )
                actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id,
                    soft=False,
                )
                logger.info(
                    "Pinning vLLM actor to node_ip=%s node_id=%s actor=%s",
                    self.pinned_node_ip,
                    node_id,
                    self.actor_name,
                )

            remote_kwargs = {
                "config": rollout_config,
                "model_config": model_config,
                "rollout_mode": RolloutMode.STANDALONE,
                "workers": [],
                "replica_rank": 0,
                "node_rank": 0,
                "gpus_per_node": total_gpus,
                "nnodes": 1,
            }
            self.server = ExtendedVLLMHttpServer.options(
                **actor_options,
            ).remote(**remote_kwargs)

            # Await Ray's ObjectRef future bridge so the API event loop remains
            # responsive during vLLM initialization.
            # Timeout scales with model size:
            # - Small models (1-2 GPUs): 300s (5 min)
            # - Medium models (4 GPUs): 600s (10 min)
            # - Large models (8+ GPUs, e.g., K2): 1800s (30 min)
            # Compute timeout based on model size (proxy: number of GPUs)
            if total_gpus >= 8:
                init_timeout = 1800  # 30 min for K2 and similar large models
            elif total_gpus >= 4:
                init_timeout = 600   # 10 min for medium MoE models
            else:
                init_timeout = 300   # 5 min for small models

            logger.info(f"Launching vLLM server (timeout={init_timeout}s for {total_gpus} GPUs)...")
            try:
                await async_get_ray_ref(self.server.launch_server.remote(), timeout_s=init_timeout)
            except SystemExit as e:
                if getattr(e, "code", None) == 15:
                    raise
                try:
                    ray_kill.kill(
                        self.server,
                        reason="vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
                self.server = None
                raise RuntimeError(f"ray.get(launch_server) triggered SystemExit for {self.actor_name}: {e}") from e
            except ray.exceptions.GetTimeoutError:
                logger.error(f"vLLM launch timed out after {init_timeout}s for {self.actor_name}")
                # Kill the stuck actor
                try:
                    ray_kill.kill(
                        self.server,
                        reason="vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
                self.server = None
                raise RuntimeError(f"vLLM actor {self.actor_name} launch timed out after {init_timeout}s")
            except Exception:
                # launch_server can fail fast on GPU placement collisions (vLLM checks free memory on startup).
                # In this case, the detached actor name is already taken; kill+remove the placement group so a
                # retry can reschedule to a different node/GPU set.
                try:
                    ray_kill.kill(
                        self.server,
                        reason="vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
                self.server = None
                raise

            self._initialized = True
            logger.info(f"MultiLoRAInferenceEngine initialized (detached actor: {self.actor_name})")

            # Register with unified model actor registry for LRU tracking
            # Include node_id for proper per-node GPU scheduling
            from mint_server.backend.actors.model_actor_inventory import ActorType
            from mint_server.backend.actors.model_actor_publication import (
                BackendModelActorLaunch,
                publish_backend_model_actor,
            )
            actor_node_id = _get_actor_node_id(self.server)
            publish_backend_model_actor(BackendModelActorLaunch(
                actor_name=self.actor_name,
                actor_type=ActorType.VLLM,
                num_gpus=total_gpus,
                actor_handle=self.server,
                namespace=PERSISTENT_NAMESPACE,
                base_model=self.model_path,
                node_id=actor_node_id,
            ))
            if actor_node_id:
                logger.info(f"vLLM actor {self.actor_name} running on node {actor_node_id[:8]}")

    async def add_lora_for_session(
        self,
        sampling_session_id: str,
        state_dict: dict,
        peft_config: dict,
    ) -> int:
        """Add frozen LoRA weights for a sampling session.

        Each call creates a new lora_int_id with frozen weights.
        Old sessions retain their weights.

        Uses tensor transfer via Ray to support distributed deployments
        where API server and inference worker have different filesystems.
        Worker saves tensors locally then creates file-based LoRARequest.
        File-based loading supports vLLM's GPU/CPU swapping.

        Auto-restarts vLLM actor if dead.

        Args:
            sampling_session_id: Unique identifier for the sampling session.
            state_dict: LoRA weight tensors (transferred via Ray object store).
            peft_config: PEFT adapter configuration dict.

        Returns:
            The allocated lora_int_id for this session.
        """
        if not self._initialized:
            await self.initialize()

        # Allocate unique ID for this session
        lora_id = await self.registry.allocate(sampling_session_id)

        # Check if we need to evict from GPU
        current_count = await self.registry.count()
        if current_count > self.max_loras:
            # vLLM handles GPU/CPU swapping automatically via max_cpu_loras
            # We just need to track in registry
            logger.debug(
                f"GPU slots full ({self.max_loras}), "
                f"vLLM will manage CPU cache for overflow"
            )

        # Transfer tensors via Ray and save locally on worker node.
        # Worker creates file-based LoRARequest for GPU/CPU swap support.
        # Auto-restart vLLM if actor died (e.g. killed for GPU allocation).
        start_time = time.time()
        traceparent = get_current_traceparent()
        print(f"[DEBUG add_lora_for_session] About to call add_lora_with_id.remote, state_dict has {len(state_dict)} keys", flush=True)
        try:
            print("[DEBUG add_lora_for_session] Calling server.add_lora_with_id.remote", flush=True)
            ref = self.server.add_lora_with_id.remote(
                lora_int_id=lora_id,
                state_dict=state_dict,
                peft_config=peft_config,
                traceparent=traceparent,
            )
            await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
            print("[DEBUG add_lora_for_session] add_lora_with_id.remote returned", flush=True)
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
            logger.warning(f"vLLM actor dead/unresponsive, reinitializing: {e}")
            print(f"[DEBUG add_lora_for_session] RayActorError/GetTimeoutError: {e}", flush=True)
            self._initialized = False
            self.server = None
            await self.initialize()
            # Retry after restart
            ref = self.server.add_lora_with_id.remote(
                lora_int_id=lora_id,
                state_dict=state_dict,
                peft_config=peft_config,
                traceparent=traceparent,
            )
            await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
        except Exception as e:
            print(f"[DEBUG add_lora_for_session] UNEXPECTED EXCEPTION: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} "
            f"(lora_int_id={lora_id}, load_time={load_time:.3f}s)"
        )
        return lora_id

    async def add_lora_for_session_from_path(
        self,
        sampling_session_id: str,
        lora_path: str,
    ) -> int:
        """Add frozen LoRA weights for a sampling session from filesystem path.

        More efficient than add_lora_for_session for large MoE models:
        - Avoids transferring 30k+ tensors through Ray object store
        - vLLM worker loads directly from shared filesystem (PFS)

        Requires shared filesystem access between API server and vLLM worker.

        Args:
            sampling_session_id: Unique identifier for the sampling session.
            lora_path: Path to saved LoRA adapter directory (must be accessible to vLLM worker).

        Returns:
            The allocated lora_int_id for this session.
        """
        if not self._initialized:
            await self.initialize()

        existing_lora_id = await self.registry.get_lora_id(sampling_session_id)
        if existing_lora_id is not None:
            logger.info(
                "LoRA already registered for session %s from path (lora_int_id=%s)",
                sampling_session_id,
                existing_lora_id,
            )
            return existing_lora_id

        # Allocate or reuse a shared LoRA ID for this adapter path. Coordinate
        # concurrent first-use requests so later sessions do not observe the
        # shared lora_int_id before the initial add_lora_from_path finishes.
        async with self._path_load_lock:
            lora_id, is_new_load = await self.registry.allocate_for_path(
                sampling_session_id, lora_path
            )
            allocated = is_new_load
            load_future = self._path_load_futures.get(lora_path)
            if is_new_load:
                loop = asyncio.get_running_loop()
                load_future = loop.create_future()
                self._path_load_futures[lora_path] = load_future

        if not is_new_load:
            if load_future is not None and not load_future.done():
                try:
                    await load_future
                except Exception:
                    await self.registry.remove_session(sampling_session_id)
                    raise
            logger.info(
                "Reused LoRA for session %s from path (lora_int_id=%s path=%s)",
                sampling_session_id,
                lora_id,
                lora_path,
            )
            return lora_id

        # Check if we need to evict from GPU
        current_count = await self.registry.count()
        if current_count > self.max_loras:
            logger.debug(
                f"GPU slots full ({self.max_loras}), "
                f"vLLM will manage CPU cache for overflow"
            )

        # Load directly from shared filesystem path
        # Much faster than Ray tensor transfer for large MoE models
        start_time = time.time()
        traceparent = get_current_traceparent()
        print(f"[DEBUG add_lora_for_session_from_path] Loading from path: {lora_path}", flush=True)

        async def _load_from_actor() -> None:
            try:
                ref = self.server.add_lora_from_path.remote(
                    lora_int_id=lora_id,
                    lora_path=lora_path,
                    lora_name=sampling_session_id,
                    traceparent=traceparent,
                )
                await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
                print("[DEBUG add_lora_for_session_from_path] add_lora_from_path.remote returned", flush=True)
            except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
                logger.warning(f"vLLM actor dead/unresponsive, reinitializing: {e}")
                print(f"[DEBUG add_lora_for_session_from_path] RayActorError/GetTimeoutError: {e}", flush=True)
                self._initialized = False
                self.server = None
                await self.initialize()
                ref = self.server.add_lora_from_path.remote(
                    lora_int_id=lora_id,
                    lora_path=lora_path,
                    lora_name=sampling_session_id,
                    traceparent=traceparent,
                )
                await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)

        try:
            await run_async_with_otel_span(
                "sampling.multi_lora.add_lora_from_path",
                _load_from_actor,
                component="multi_lora_engine",
                op="sampling.add_lora_from_path",
                request_id=sampling_session_id,
                attributes={
                    "actor_name": self.actor_name,
                    "sampling_session_id": sampling_session_id,
                },
            )
        except Exception as e:
            print(f"[DEBUG add_lora_for_session_from_path] UNEXPECTED EXCEPTION: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            if allocated:
                try:
                    await self.registry.remove_session(sampling_session_id)
                except Exception as cleanup_e:
                    logger.warning(
                        "Failed to roll back session=%s after add_lora failure: %s: %s",
                        sampling_session_id,
                        type(cleanup_e).__name__,
                        cleanup_e,
                    )
            if load_future is not None and not load_future.done():
                load_future.set_exception(e)
            async with self._path_load_lock:
                self._path_load_futures.pop(lora_path, None)
            raise
        if load_future is not None and not load_future.done():
            load_future.set_result(None)
        async with self._path_load_lock:
            self._path_load_futures.pop(lora_path, None)
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} from path "
            f"(lora_int_id={lora_id}, load_time={load_time:.3f}s)"
        )
        return lora_id

    async def generate(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
        stop: object | None = None,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
    ) -> GenerateResult:
        """Generate tokens using session-specific LoRA or base model.

        If sampling_session_id is None or has no registered LoRA, uses base model.

        Args:
            sampling_session_id: The sampling session to use, or None for base model.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            logprobs: Whether to return log probabilities.

        Returns:
            GenerateResult with generated tokens and metadata.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
        traceparent = get_current_traceparent()

        async def _dispatch_generate() -> dict:
            if lora_id is not None:
                logger.info(
                    "[generate dispatch] req=%s actor=%s lora_id=%s prompt_len=%d max_tokens=%d",
                    request_id, self.actor_name, lora_id, len(prompt_ids), max_tokens,
                )
                ref = self.server.generate_with_lora.remote(
                    prompt_ids=prompt_ids,
                    request_id=request_id,
                    lora_int_id=lora_id,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    logprobs=logprobs,
                    traceparent=traceparent,
                )
            else:
                logger.info(
                    "[generate dispatch] req=%s actor=%s base_model prompt_len=%d max_tokens=%d",
                    request_id, self.actor_name, len(prompt_ids), max_tokens,
                )
                ref = self.server.generate_base.remote(
                    prompt_ids=prompt_ids,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    logprobs=logprobs,
                    traceparent=traceparent,
                )

            t0_ray = time.time()
            result = await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name, request_id=request_id)
            ray_elapsed = time.time() - t0_ray
            if ray_elapsed > 30.0:
                logger.warning(
                    "[generate slow] req=%s actor=%s ray_get_elapsed=%.1fs",
                    request_id, self.actor_name, ray_elapsed,
                )
            return result

        result = await run_async_with_otel_span(
            "sampling.multi_lora.generate",
            _dispatch_generate,
            component="multi_lora_engine",
            op="sampling.generate",
            request_id=request_id,
            attributes={
                "actor_name": self.actor_name,
                "sampling_session_id": sampling_session_id,
                "prompt_tokens": len(prompt_ids),
                "max_tokens": max_tokens,
                "num_samples": 1,
                "lora_id": lora_id,
            },
        )

        timing_total_s = result.get("_timing_total_s")
        if timing_total_s is not None:
            try:
                total_s = float(timing_total_s)
            except (TypeError, ValueError):
                total_s = None
            if total_s is not None:
                first_tok_s = result.get("_timing_first_tok_s")
                lora_tag = f"lora_id={lora_id}" if lora_id is not None else "lora_id=None"
                logger.info(
                    f"[vLLM timing] generate req={request_id} prompt_len={len(prompt_ids)} "
                    f"max_tokens={max_tokens} {lora_tag} total_s={total_s:.3f} first_tok_s={first_tok_s}"
                )

        return GenerateResult(
            token_ids=result["token_ids"],
            logprobs=result.get("logprobs"),
            stop_reason=result.get("stop_reason"),
            routed_experts=result.get("routed_experts"),
            timing_total_s=_float_or_none(result.get("_timing_total_s")),
            timing_first_tok_s=_float_or_none(result.get("_timing_first_tok_s")),
        )

    async def generate_many(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        num_samples: int,
        max_tokens: int,
        stop: object | None = None,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
    ) -> list[GenerateResult]:
        """Generate multiple sequences for the same prompt in a single vLLM request."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1 (got {num_samples})")

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
        traceparent = get_current_traceparent()

        async def _dispatch_generate_many():
            if lora_id is not None:
                ref = self.server.generate_with_lora.remote(
                    prompt_ids=prompt_ids,
                    request_id=request_id,
                    lora_int_id=lora_id,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    logprobs=logprobs,
                    n=num_samples,
                    traceparent=traceparent,
                )
            else:
                ref = self.server.generate_base.remote(
                    prompt_ids=prompt_ids,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    logprobs=logprobs,
                    n=num_samples,
                    traceparent=traceparent,
                )
            return await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)

        raw = await run_async_with_otel_span(
            "sampling.multi_lora.generate_many",
            _dispatch_generate_many,
            component="multi_lora_engine",
            op="sampling.generate_many",
            request_id=request_id,
            attributes={
                "actor_name": self.actor_name,
                "sampling_session_id": sampling_session_id,
                "prompt_tokens": len(prompt_ids),
                "max_tokens": max_tokens,
                "num_samples": num_samples,
                "lora_id": lora_id,
            },
        )
        logger.info(
            "[generate_many result] req=%s actor=%s type=%s",
            request_id,
            self.actor_name,
            type(raw).__name__,
        )

        if isinstance(raw, dict):
            raw_list: list[dict] = [raw]
        else:
            raw_list = list(raw)
        logger.info(
            "[generate_many result] req=%s actor=%s result_count=%d",
            request_id,
            self.actor_name,
            len(raw_list),
        )

        if raw_list:
            timing_total_s = raw_list[0].get("_timing_total_s")
            if timing_total_s is not None:
                try:
                    total_s = float(timing_total_s)
                except (TypeError, ValueError):
                    total_s = None
                if total_s is not None:
                    first_tok_s = raw_list[0].get("_timing_first_tok_s")
                    lora_tag = f"lora_id={lora_id}" if lora_id is not None else "lora_id=None"
                    logger.info(
                        f"[vLLM timing] generate_many req={request_id} prompt_len={len(prompt_ids)} "
                        f"max_tokens={max_tokens} num_samples={num_samples} {lora_tag} "
                        f"total_s={total_s:.3f} first_tok_s={first_tok_s}"
                    )

        return [
            GenerateResult(
                token_ids=r["token_ids"],
                logprobs=r.get("logprobs"),
                stop_reason=r.get("stop_reason"),
                routed_experts=r.get("routed_experts"),
                timing_total_s=_float_or_none(r.get("_timing_total_s")),
                timing_first_tok_s=_float_or_none(r.get("_timing_first_tok_s")),
            )
            for r in raw_list
        ]

    async def compute_logprobs(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
    ) -> list[float | None]:
        """Compute logprobs using session-specific LoRA or base model.

        If sampling_session_id is None or has no registered LoRA, uses base model.

        Returns a list of length len(prompt_ids), where:
        - logprobs[0] is None (first token has no conditioning context)
        - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1

        Args:
            sampling_session_id: The sampling session to use, or None for base model.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.

        Returns:
            List of logprobs, length = len(prompt_ids).
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
        traceparent = get_current_traceparent()

        async def _dispatch_compute_logprobs():
            if lora_id is not None:
                try:
                    ref = self.server.compute_prompt_logprobs_with_lora.remote(
                        prompt_ids=prompt_ids,
                        request_id=request_id,
                        lora_int_id=lora_id,
                        traceparent=traceparent,
                    )
                    return await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    if any(s in msg for s in ("OutOfMemoryError", "CUDA out of memory", "EngineDeadError")):
                        logger.error(f"vLLM compute_logprobs failed; killing actor {self.actor_name}: {msg}")
                        try:
                            from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

                            ray_kill.kill(
                                self.server,
                                reason="vllm_compute_logprobs_failed",
                                actor_name=self.actor_name,
                                namespace=PERSISTENT_NAMESPACE,
                                no_restart=True,
                            )
                            get_model_actor_supervisor().unregister(self.actor_name)
                        except Exception:
                            pass
                        self.server = None
                        self._initialized = False
                    raise
            try:
                ref = self.server.compute_prompt_logprobs_base.remote(
                    prompt_ids=prompt_ids,
                    request_id=request_id,
                    traceparent=traceparent,
                )
                return await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if any(s in msg for s in ("OutOfMemoryError", "CUDA out of memory", "EngineDeadError")):
                    logger.error(f"vLLM compute_logprobs failed; killing actor {self.actor_name}: {msg}")
                    try:
                        from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

                        ray_kill.kill(
                            self.server,
                            reason="vllm_compute_logprobs_failed",
                            actor_name=self.actor_name,
                            namespace=PERSISTENT_NAMESPACE,
                            no_restart=True,
                        )
                        get_model_actor_supervisor().unregister(self.actor_name)
                    except Exception:
                        pass
                    self.server = None
                    self._initialized = False
                raise

        result = await run_async_with_otel_span(
            "sampling.multi_lora.compute_logprobs",
            _dispatch_compute_logprobs,
            component="multi_lora_engine",
            op="sampling.compute_logprobs",
            request_id=request_id,
            attributes={
                "actor_name": self.actor_name,
                "sampling_session_id": sampling_session_id,
                "prompt_tokens": len(prompt_ids),
                "lora_id": lora_id,
            },
        )

        return list(result)

    async def compute_topk(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        k: int = 10,
    ) -> list[list[tuple[int, float]] | None]:
        """Compute top-K tokens using session-specific LoRA or base model.

        If sampling_session_id is None or has no registered LoRA, uses base model.

        Returns a list of length len(prompt_ids), where:
        - topk[0] is None (first token has no conditioning context)
        - topk[i] is a list of (token_id, logprob) pairs for token i's distribution (i >= 1)

        Args:
            sampling_session_id: The sampling session to use, or None for base model.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.
            k: Number of top tokens to return.

        Returns:
            List of per-position top-k lists, length = len(prompt_ids).
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
        traceparent = get_current_traceparent()

        if lora_id is not None:
            # Compute top-K with session-specific LoRA
            ref = self.server.compute_prompt_topk_with_lora.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                lora_int_id=lora_id,
                k=k,
                traceparent=traceparent,
            )
        else:
            # Compute top-K with base model (no LoRA)
            ref = self.server.compute_prompt_topk_base.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                k=k,
                traceparent=traceparent,
            )

        result = await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)

        return list(result)

    async def remove_session(self, sampling_session_id: str) -> bool:
        """Remove a sampling session and its LoRA.

        Args:
            sampling_session_id: The session to remove.

        Returns:
            True if session was removed, False if not found.
        """
        lora_id = await self.registry.get_lora_id(sampling_session_id)
        if lora_id is None:
            return False

        lora_id, should_unload = await self.registry.remove_session(sampling_session_id)
        if lora_id is None:
            return False

        if should_unload:
            try:
                ref = self.server.remove_lora.remote(lora_id)
                await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
            except Exception as e:
                logger.warning(f"Failed to remove LoRA {lora_id} from engine: {e}")

        logger.info(
            "Removed session %s (lora_int_id=%s should_unload=%s)",
            sampling_session_id,
            lora_id,
            should_unload,
        )
        return True

    async def restore_loaded_session(
        self,
        *,
        sampling_session_id: str,
        adapter_path: str,
        lora_int_id: int,
    ) -> int:
        """Restore a detached control-plane mapping for an already-loaded LoRA."""
        return await self.registry.restore_existing_session(
            sampling_session_id,
            adapter_path=adapter_path,
            lora_int_id=lora_int_id,
        )

    async def shutdown(self, kill_actor: bool = False) -> None:
        """Disconnect from the engine (optionally kill the persistent actor).

        Args:
            kill_actor: If True, kill the persistent vLLM actor.
                        If False (default), just disconnect - actor survives for reuse.
        """
        if self.server is not None and kill_actor:
            try:
                ray_kill.kill(
                    self.server,
                    reason="vllm_shutdown",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                )
                logger.info("Killed persistent vLLM actor")
            except Exception as e:
                logger.warning(f"Error killing server actor: {e}")
            try:
                remove_named_placement_group(f"{self.actor_name}_pg", namespace=PERSISTENT_NAMESPACE)
            except Exception:
                pass
        self.server = None
        self._initialized = False
        logger.info("MultiLoRAInferenceEngine disconnected")


def _model_to_actor_name(model_name: str) -> str:
    """Convert model name to a valid Ray actor name.

    Examples:
        "Qwen/Qwen2.5-7B-Instruct" -> "mint_vllm_qwen2.5-7b-instruct"
        "Qwen/Qwen3-30B-A3B-Instruct-2507" -> "mint_vllm_qwen3-30b-a3b-instruct-2507"
    """
    # Extract model part after "/"
    if "/" in model_name:
        model_part = model_name.split("/")[-1]
    else:
        model_part = model_name
    # Lowercase and replace invalid chars
    safe_name = model_part.lower().replace(" ", "_")
    return f"mint_vllm_{safe_name}"


def _resolve_model_path(model_name: str) -> str:
    """Resolve model name to a concrete local snapshot path on PFS."""
    # Map of model names to local paths
    model_paths = {
        # Dense models
        "Qwen/Qwen2.5-7B-Instruct": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "Qwen/Qwen3-0.6B": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "Qwen/Qwen3-4B-Instruct-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/main",
        "Qwen/Qwen3-4B-Thinking-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/main",
        # MoE models (all share same architecture, different checkpoints)
        "Qwen/Qwen3-30B-A3B-Instruct-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "Qwen/Qwen3-30B-A3B": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/main",
        "Qwen/Qwen3-30B-A3B-Base": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/main",
        "Qwen/Qwen3-30B-A3B-Thinking-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/main",
        "Qwen/Qwen3-235B-A22B-Instruct-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e",
        "Qwen/Qwen3-235B-A22B-Thinking-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5",
        # K2 models (1T params MoE, requires FP8)
        "moonshotai/Kimi-K2-Instruct": "/vePFS-Mindverse/share/huggingface/hub/models--unsloth--Kimi-K2-Instruct-0905-BF16/snapshots/fbaf30b3baf5fdc2b2170ae04f4ff4948b0487cb",
        "moonshotai/Kimi-K2-Thinking": "/vePFS-Mindverse/share/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd",
    }

    resolved = model_paths.get(model_name)
    if resolved is None:
        if model_name == "Qwen/Qwen3.5-27B":
            try:
                from mint_server.backend.inference.qwen35_text_vllm_adapter import resolve_hf_config_dir

                qwen35_path = resolve_hf_config_dir(model_name)
            except Exception:
                logger.debug("Failed to resolve Qwen3.5 HF cache path", exc_info=True)
                qwen35_path = None
            if qwen35_path:
                return qwen35_path
        return model_name

    resolved_path = Path(resolved)
    if resolved_path.exists():
        return str(resolved_path.resolve())

    snapshots_dir = resolved_path.parent
    if snapshots_dir.name == "snapshots" and snapshots_dir.exists():
        snapshot_dirs = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshot_dirs:
            return str(snapshot_dirs[-1].resolve())

    # Fall back to the configured value so callers still get the original path in logs.
    return resolved

class MultiModelInferenceManager:
    """Manages multiple vLLM engines, one per base model.

    Provides dynamic engine creation based on model name.
    Each base model gets its own persistent vLLM actor.
    """

    def __init__(
        self,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        max_loras: int = DEFAULT_MAX_LORAS,
        max_cpu_loras: int = DEFAULT_MAX_CPU_LORAS,
        max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
    ):
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_lora_rank = max_lora_rank

        # Dict of model_name -> engine
        self._engines: dict[str, MultiLoRAInferenceEngine] = {}
        # Per-model init locks: avoid blocking other models while one
        # model's vLLM actor is (re)initializing.
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _get_model_lock(self, model_name: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(model_name)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[model_name] = lock
            return lock

    async def _named_single_node_actor_is_reusable(self, actor_name: str) -> bool:
        """Return True only when the detached actor is usable for immediate reuse."""
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
        except ValueError:
            return False
        except ray.exceptions.RayActorError:
            return False

        try:
            await async_get_ray_ref(actor.__ray_ready__.remote(), timeout_s=5)
            ready_method_name = None
            ready_probe = None
            for method_name in ("is_engine_ready", "is_ready"):
                candidate = getattr(actor, method_name, None)
                if candidate is None:
                    continue
                ready_method_name = method_name
                ready_probe = candidate
                break
            if ready_probe is None:
                logger.warning(
                    "named actor probe found no readiness RPC actor=%s; reusing actor after __ray_ready__",
                    actor_name,
                )
                return True
            engine_ready = await async_get_ray_ref(ready_probe.remote(), timeout_s=10)
            logger.debug(
                "named actor probe readiness actor=%s method=%s ready=%s",
                actor_name,
                ready_method_name,
                engine_ready,
            )
            return bool(engine_ready)
        except SystemExit as e:
            if getattr(e, "code", None) == 15:
                raise
            logger.warning(
                "named actor probe hit SystemExit actor=%s err=%s; treating as non-reusable",
                actor_name,
                e,
            )
            return False
        except ray.exceptions.GetTimeoutError:
            # Busy actors still occupy the pinned node and should be reused rather than
            # tripping a false capacity failure before initialize() reconnects to them.
            logger.info(
                "named actor probe timed out actor=%s; treating existing actor as reusable",
                actor_name,
            )
            return True
        except ray.exceptions.RayActorError:
            logger.warning(
                "named actor probe found stale/dead actor=%s; will run capacity check before recreate",
                actor_name,
            )
            return False

    async def get_engine(self, model_name: str) -> MultiLoRAInferenceEngine:
        """Get or create engine for a model.

        If cached engine exists, verifies actor is alive before returning.
        If actor is dead, removes from cache and creates new engine.

        For models requiring TP > 8 (multi-node), uses MultiNodeInferenceEngine
        which leverages vLLM's native Ray distributed backend.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")

        Returns:
            Initialized inference engine for the model.
        """
        lock = await self._get_model_lock(model_name)
        async with lock:
            logger.info("get_engine model=%s stage=entered_lock", model_name)
            if model_name in self._engines:
                engine = self._engines[model_name]
                # Check if actor is still alive before returning cached engine
                # Handle both MultiLoRAInferenceEngine and MultiNodeInferenceEngine
                actor_handle = getattr(engine, 'server', None) or getattr(engine, 'engine', None)
                if actor_handle is not None:
                    if not hasattr(engine, 'server'):
                        # MultiNodeInferenceEngine actors serialize methods. Probing `is_ready()` here
                        # can itself queue behind long-running generations and wedge unrelated
                        # sessions before they even reach `after_get_engine`. Use the detached
                        # actor name lookup instead: it is cheap, avoids queuing on the actor,
                        # and still lets us drop dead cached handles before reuse.
                        from mint_server.backend.inference.multinode_inference import (
                            PERSISTENT_NAMESPACE as MULTINODE_PERSISTENT_NAMESPACE,
                        )

                        actor_name = getattr(engine, "actor_name", None)
                        if actor_name:
                            try:
                                ray.get_actor(actor_name, namespace=MULTINODE_PERSISTENT_NAMESPACE)
                            except (ValueError, ray.exceptions.RayActorError):
                                logger.warning(
                                    "Cached multi-node vLLM engine for %s has no live named actor, recreating",
                                    model_name,
                                )
                                self._engines.pop(model_name, None)
                                _invalidate_model_session_loras(model_name)
                            else:
                                logger.info("get_engine model=%s stage=return_cached_engine_multinode", model_name)
                                return engine
                        else:
                            logger.info("get_engine model=%s stage=return_cached_engine_multinode", model_name)
                            return engine
                    try:
                        if hasattr(engine, 'server'):
                            await async_get_ray_ref(actor_handle.__ray_ready__.remote(), timeout_s=5)
                        else:
                            await async_get_ray_ref(actor_handle.is_ready.remote(), timeout_s=5)
                        # Actor alive, return cached engine
                        logger.info("get_engine model=%s stage=return_cached_engine", model_name)
                        return engine
                    except SystemExit as e:
                        if getattr(e, "code", None) == 15:
                            raise
                        logger.warning(
                            f"Cached vLLM engine for {model_name} hit SystemExit during is_alive check; recreating"
                        )
                        self._engines.pop(model_name, None)
                        _invalidate_model_session_loras(model_name)
                    except ray.exceptions.GetTimeoutError:
                        # Actor tasks can queue behind long-running generations/logprobs.
                        # A short timeout here is not evidence of death.
                        logger.warning(
                            f"Cached vLLM engine for {model_name} is busy during is_alive check; reusing cached engine"
                        )
                        return engine
                    except ray.exceptions.RayActorError:
                        # Actor is dead, remove from cache and recreate
                        logger.warning(
                            f"Cached vLLM engine for {model_name} has dead actor, recreating"
                        )
                        self._engines.pop(model_name, None)
                        _invalidate_model_session_loras(model_name)
                else:
                    # Engine has no actor handle, remove stale entry
                    self._engines.pop(model_name, None)
                    _invalidate_model_session_loras(model_name)

            # Get model config for parallelism settings
            from mint_server.backend.core.model_registry import get_model_config
            config = get_model_config(model_name)

            # Create unique actor name for this model
            actor_name = _model_to_actor_name(model_name)
            model_path = _resolve_model_path(model_name)

            existing_named_actor = False
            try:
                if not ray.is_initialized():
                    init_ray(
                        address="auto",
                        namespace=PERSISTENT_NAMESPACE,
                        ignore_reinit_error=True,
                    )
                existing_named_actor = await self._named_single_node_actor_is_reusable(actor_name)
                if existing_named_actor:
                    logger.info(
                        "get_engine model=%s stage=existing_named_actor_found actor=%s namespace=%s",
                        model_name,
                        actor_name,
                        PERSISTENT_NAMESPACE,
                    )
            except Exception as e:
                logger.debug(
                    "get_engine model=%s stage=existing_named_actor_probe_failed actor=%s err=%s",
                    model_name,
                    actor_name,
                    e,
                )

            total_gpus = int(config.inference_tp) * int(config.inference_dp)
            pinned_node_ip: str | None = None
            raw_vllm_placement = _read_process_env_var("MINT_VLLM_MODEL_PLACEMENT_JSON")
            raw_model_placement = _read_process_env_var("MINT_MODEL_PLACEMENT_JSON")
            lookup_keys = [
                str(model_name).strip(),
                str(model_name).strip().lower(),
                str(actor_name).strip(),
                str(actor_name).strip().lower(),
            ]
            context = f"single_node_vllm_pin model={model_name!r} actor={actor_name!r}"
            if config.vllm_engine == "async":
                logger.info(
                    "single_node_vllm_pin_lookup model=%r actor=%r skipped_for_async_engine=True raw_vllm_placement_present=%s raw_model_placement_present=%s",
                    model_name,
                    actor_name,
                    bool(raw_vllm_placement),
                    bool(raw_model_placement),
                )
            else:
                placement = parse_model_gpu_placement(
                    raw_json=raw_vllm_placement,
                    lookup_keys=lookup_keys,
                    env_var_name="MINT_VLLM_MODEL_PLACEMENT_JSON",
                    context=context,
                    replica=0,
                )
                if placement is None:
                    placement = parse_model_gpu_placement(
                        raw_json=raw_model_placement,
                        lookup_keys=lookup_keys,
                        env_var_name="MINT_MODEL_PLACEMENT_JSON",
                        context=context,
                        replica=0,
                    )
                if placement is not None:
                    if len(placement.slices) != 1:
                        raise RuntimeError(
                            f"{context}: single-node vLLM requires exactly 1 placement slice, got {len(placement.slices)}"
                        )
                    if placement.total_gpus != int(total_gpus):
                        raise RuntimeError(
                            f"{context}: placement GPU count mismatch, need {total_gpus} GPUs, got {placement.total_gpus}"
                        )
                    pinned_node_ip = placement.slices[0].node_ip
                    logger.info(
                        "single_node_vllm_pin_lookup model=%r actor=%r pinned_node_ip=%s lookup_keys=%s placement_slices=%s raw_vllm_placement_present=%s raw_model_placement_present=%s",
                        model_name,
                        actor_name,
                        pinned_node_ip,
                        lookup_keys,
                        [
                            {
                                "replica": slice_.replica,
                                "worker_index": slice_.worker_index,
                                "gpu_count": slice_.gpu_count,
                                "node_ip": slice_.node_ip,
                            }
                            for slice_ in placement.slices
                        ],
                        bool(raw_vllm_placement),
                        bool(raw_model_placement),
                    )
                else:
                    logger.info(
                        "single_node_vllm_pin_lookup model=%r actor=%r pinned_node_ip=None lookup_keys=%s raw_vllm_placement_present=%s raw_model_placement_present=%s",
                        model_name,
                        actor_name,
                        lookup_keys,
                        bool(raw_vllm_placement),
                        bool(raw_model_placement),
                    )

            # Determine quantization from model config (None = vLLM auto-detect from config.json)
            quantization = config.quantization
            if pinned_node_ip is not None and existing_named_actor:
                logger.info(
                    "get_engine model=%s stage=skip_capacity_check_existing_named_actor actor=%s pinned_node_ip=%s",
                    model_name,
                    actor_name,
                    pinned_node_ip,
                )
            elif pinned_node_ip is not None:
                assert_node_ip_capacity(
                    required_gpus_by_node_ip={pinned_node_ip: total_gpus},
                    context=f"single_node_vllm_pin model={model_name!r} actor={actor_name!r}",
                )

            # Determine max_loras: per-model override > MoE default (1) > global default
            model_max_loras_override = config.max_loras
            if model_max_loras_override is not None:
                model_max_loras = model_max_loras_override
            elif config.is_moe:
                model_max_loras = 1  # Default for MoE: minimal LoRA support
            else:
                model_max_loras = self.max_loras

            # Determine max_cpu_loras:
            # - per-model override (if set)
            # - LoRA disabled -> 0
            # - MoE default -> vLLM default (None => max_cpu_loras=max_loras)
            # - dense default -> global default
            model_max_cpu_loras_override = getattr(config, "max_cpu_loras", None)
            if model_max_cpu_loras_override is not None:
                model_max_cpu_loras: int | None = int(model_max_cpu_loras_override)
            elif model_max_loras == 0:
                model_max_cpu_loras = 0
            elif config.is_moe:
                model_max_cpu_loras = None
            else:
                model_max_cpu_loras = self.max_cpu_loras

            # Use per-model max_lora_rank override if specified (K2 needs rank 8 to fit).
            # Large MoE models have huge LoRA buffers: 384 experts × rank × hidden_size.
            model_max_lora_rank = config.max_lora_rank or self.max_lora_rank

            # Use per-model gpu_memory_utilization if specified (K2 needs lower for LoRA headroom)
            model_gpu_util = config.gpu_memory_utilization or self.gpu_memory_utilization

            # Use per-model max_model_len if specified (K2 needs 128K, default 262K exceeds KV cache)
            model_max_model_len = config.max_model_len or self.max_model_len

            # Use per-model max_num_seqs if specified (K2 needs reduced value for KV cache)
            model_max_num_seqs = config.max_num_seqs or 256  # Default: 256

            # Use per-model kv_cache_dtype if specified (FP8 KV cache halves memory)
            model_kv_cache_dtype = config.kv_cache_dtype

            if int(config.total_gpus) > 8 and config.vllm_engine != "async":
                raise RuntimeError(
                    f"Multi-node vLLM requires vllm_engine=async (model={model_name}, vllm_engine={config.vllm_engine})"
                )

            if config.vllm_engine == "async":
                from mint_server.backend.inference.multinode_inference import MultiNodeInferenceEngine

                total_gpus = int(config.total_gpus)
                pipeline_parallel_size = int(getattr(config, "inference_pp", 1) or 1)
                enable_expert_parallel = _env_flag(
                    "MINT_VLLM_ENABLE_EXPERT_PARALLEL",
                    default=bool(config.is_moe and config.inference_dp > 1),
                )
                logger.info(
                    f"Creating multi-node vLLM engine for model {model_name}: "
                    f"actor={actor_name}, TP={config.inference_tp}, PP={pipeline_parallel_size}, DP={config.inference_dp}, "
                    f"backend={config.vllm_distributed_executor_backend}, expert_parallel={enable_expert_parallel}, "
                    f"total_gpus={total_gpus}, gpu_util={model_gpu_util}, quant={quantization}, "
                    f"max_loras={model_max_loras}, max_lora_rank={model_max_lora_rank}, "
                    f"max_model_len={model_max_model_len}, max_num_seqs={model_max_num_seqs}, "
                    f"kv_cache_dtype={model_kv_cache_dtype}"
                )

                engine = MultiNodeInferenceEngine(
                    model_path=model_path,
                    model_name=model_name,
                    tensor_parallel_size=config.inference_tp,
                    pipeline_parallel_size=pipeline_parallel_size,
                    data_parallel_size=config.inference_dp,
                    enable_expert_parallel=enable_expert_parallel,
                    gpu_memory_utilization=model_gpu_util,
                    max_model_len=model_max_model_len,
                    max_loras=model_max_loras,
                    max_cpu_loras=model_max_cpu_loras,
                    max_lora_rank=model_max_lora_rank,
                    max_num_seqs=model_max_num_seqs,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    quantization=quantization,
                    kv_cache_dtype=model_kv_cache_dtype,
                    actor_name=actor_name,
                    distributed_executor_backend=config.vllm_distributed_executor_backend,
                )
            else:
                # Use standard MultiLoRAInferenceEngine for single-node models
                logger.info(
                    f"Creating vLLM engine for model {model_name}: "
                    f"actor={actor_name}, TP={config.inference_tp}, DP={config.inference_dp}, "
                    f"quant={quantization}, max_loras={model_max_loras}, max_lora_rank={model_max_lora_rank}"
                )

                engine = MultiLoRAInferenceEngine(
                    model_path=model_path,
                    model_name=model_name,
                    tensor_parallel_size=config.inference_tp,
                    data_parallel_size=config.inference_dp,
                    gpu_memory_utilization=model_gpu_util,
                    max_model_len=model_max_model_len,
                    max_num_seqs=model_max_num_seqs,
                    max_loras=model_max_loras,
                    max_cpu_loras=self.max_cpu_loras if model_max_cpu_loras is None else int(model_max_cpu_loras),
                    max_lora_rank=model_max_lora_rank,
                    pinned_node_ip=pinned_node_ip,
                    actor_name=actor_name,
                    quantization=quantization,
                )

            logger.info("get_engine model=%s stage=before_engine_initialize", model_name)
            await engine.initialize()
            logger.info("get_engine model=%s stage=after_engine_initialize", model_name)

            self._engines[model_name] = engine
            from mint_server.backend.core.model_registry import is_topology_desired_model

            if is_topology_desired_model(model_name):
                from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

                model_actor_supervisor = get_model_actor_supervisor()
                if not model_actor_supervisor.set_protected(actor_name, True):
                    logger.warning(
                        f"Failed to protect vLLM actor for topology desired model {model_name}: actor={actor_name}"
                    )
            logger.info(f"Engine created for {model_name}")
            return engine

    def get_engine_if_exists(self, model_name: str) -> MultiLoRAInferenceEngine | None:
        """Get engine for model if already created, None otherwise."""
        return self._engines.get(model_name)

    async def shutdown_all(self, kill_actors: bool = False) -> None:
        """Shutdown all engines.

        Args:
            kill_actors: If True, kill the persistent actors. If False, just disconnect.
        """
        for model_name, engine in self._engines.items():
            logger.info(f"Shutting down engine for {model_name}")
            await engine.shutdown(kill_actor=kill_actors)
        self._engines.clear()

    def list_models(self) -> list[str]:
        """List all models with active engines."""
        return list(self._engines.keys())


def _list_named_vllm_actor_names(*, namespace: str) -> list[str]:
    """List vLLM actor names by Ray named actor registry.

    This is useful for detecting supervisor inventory mismatches.
    Callers must not silently treat this registry as authoritative.
    """
    names: list[str] = []
    for a in ray.util.list_named_actors(all_namespaces=True):
        if a.get("namespace") != namespace:
            continue
        name = a.get("name") or ""
        if not name:
            continue
        if name.startswith("mint_vllm_"):
            names.append(name)
    return sorted(set(names))


def kill_persistent_vllm_actor(model_name: str | None = None) -> bool:
    """Kill persistent vLLM actor(s).

    Args:
        model_name: If provided, kill actor for this specific model.
                   If None, kill ALL vLLM actors.

    Returns:
        True if any actor was killed, False if none found.

    Raises:
        ModelActorSupervisorStaleError: When supervisor inventory reports no vLLM actors but Ray named
            actors exist in the namespace. This is an accounting invariant violation
            and must be surfaced rather than falling back silently.
    """
    from mint_server.backend.actors.model_actor_supervisor import ActorType, get_model_actor_supervisor

    if not ray.is_initialized():
        init_ray(
            namespace=PERSISTENT_NAMESPACE,
            ignore_reinit_error=True,
        )

    model_actor_supervisor = get_model_actor_supervisor()

    if model_name:
        # Kill specific model's actor
        actor_name = _model_to_actor_name(model_name)
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            ray_kill.kill(
                actor,
                reason="vllm_kill_by_name",
                actor_name=actor_name,
                namespace=PERSISTENT_NAMESPACE,
            )
            logger.info(f"Killed vLLM actor: {actor_name}")
            model_actor_supervisor.unregister(actor_name)
            try:
                remove_named_placement_group(f"{actor_name}_pg", namespace=PERSISTENT_NAMESPACE)
            except Exception:
                pass
            return True
        except ValueError:
            logger.info(f"No vLLM actor found: {actor_name}")
            try:
                remove_named_placement_group(f"{actor_name}_pg", namespace=PERSISTENT_NAMESPACE)
            except Exception:
                pass
            return False
    else:
        # Kill ALL vLLM actors via model actor registry
        killed_any = False
        for entry in model_actor_supervisor.iter_entries():
            if entry.actor_type == ActorType.VLLM:
                try:
                    actor = ray.get_actor(entry.actor_name, namespace=entry.namespace)
                    ray_kill.kill(
                        actor,
                        reason="vllm_kill_by_name",
                        actor_name=entry.actor_name,
                        namespace=entry.namespace,
                    )
                    logger.info(f"Killed vLLM actor: {entry.actor_name}")
                    model_actor_supervisor.unregister(entry.actor_name)
                    try:
                        remove_named_placement_group(
                            f"{entry.actor_name}_pg",
                            namespace=entry.namespace,
                        )
                    except Exception:
                        pass
                    killed_any = True
                except ValueError:
                    logger.warning(f"vLLM actor not found in Ray: {entry.actor_name}")
                    model_actor_supervisor.unregister(entry.actor_name)
                    try:
                        remove_named_placement_group(
                            f"{entry.actor_name}_pg",
                            namespace=entry.namespace,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"Error killing vLLM actor {entry.actor_name}: {e}")
        if not killed_any:
            from mint_server.backend.actors.model_actor_supervisor import ModelActorSupervisorStaleError

            named = _list_named_vllm_actor_names(namespace=PERSISTENT_NAMESPACE)
            if named:
                raise ModelActorSupervisorStaleError(
                    "ModelActorSupervisor inventory lists 0 vLLM actors but Ray named-actor registry has "
                    f"{len(named)} vLLM actor(s) in namespace={PERSISTENT_NAMESPACE!r}: {named!r}"
                )
            logger.info("No vLLM actors found in ModelActorSupervisor inventory")
        return killed_any


def check_persistent_vllm_actor(model_name: str | None = None) -> bool:
    """Check if vLLM actor(s) exist and are alive.

    Args:
        model_name: If provided, check for this specific model's actor.
                   If None, check for ANY running vLLM actor.

    Returns:
        True if matching actor exists and is alive.
    """
    from mint_server.backend.actors.model_actor_supervisor import ActorType, get_model_actor_supervisor

    if not ray.is_initialized():
        init_ray(
            namespace=PERSISTENT_NAMESPACE,
            ignore_reinit_error=True,
        )

    def _is_actor_ready(actor) -> bool:
        """Return True if actor responds and (if supported) its engine is healthy.

        Notes:
        - For verl's ExtendedVLLMHttpServer: prefer is_engine_ready()
        - For MultiNodeVLLMEngine: prefer is_ready() (touches EngineCore)
        - If the readiness call times out, treat readiness as unknown and fail closed.
        """
        try:
            try:
                ref = actor.is_engine_ready.remote()
            except AttributeError:
                ref = actor.is_ready.remote()
        except AttributeError:
            ref = actor.__ray_ready__.remote()

        try:
            res = ray.get(ref, timeout=5)
        except ray.exceptions.GetTimeoutError:
            return False

        if isinstance(res, bool):
            return res
        return True

    if model_name:
        # Check specific model's actor
        actor_name = _model_to_actor_name(model_name)
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            return _is_actor_ready(actor)
        except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
            return False
    else:
        # Check for ANY vLLM actor via model actor registry
        model_actor_supervisor = get_model_actor_supervisor()
        for entry in model_actor_supervisor.iter_entries():
            if entry.actor_type == ActorType.VLLM:
                try:
                    actor = ray.get_actor(entry.actor_name, namespace=entry.namespace)
                    if _is_actor_ready(actor):
                        return True
                except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                    continue
        return False


def list_vllm_actors() -> list[dict]:
    """List all vLLM actors from model actor registry.

    Returns:
        List of actor info dicts with name, gpus, base_model.
    """
    from mint_server.backend.actors.model_actor_supervisor import ActorType, get_model_actor_supervisor

    model_actor_supervisor = get_model_actor_supervisor()
    out = [
        {"name": e.actor_name, "gpus": e.num_gpus, "base_model": e.base_model}
        for e in model_actor_supervisor.iter_entries()
        if e.actor_type == ActorType.VLLM
    ]
    seen = {a["name"] for a in out}
    for name in _list_named_vllm_actor_names(namespace=PERSISTENT_NAMESPACE):
        if name in seen:
            continue
        out.append({"name": name, "gpus": 0, "base_model": ""})
    return out


# Global instance (initialized in app lifespan)
multi_lora_engine: MultiLoRAInferenceEngine | None = None
