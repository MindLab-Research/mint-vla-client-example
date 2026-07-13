"""Non-Ray (in-process) runtime factory for OpenPI pi0.5 training.

Step 1 of the `Openpi-LoRA-Separate` effort: detach pi0.5 from the Ray runtime.

The default pi0.5 runtime factory (`openpi_pi05_training._default_runtime_factory`)
routes every worker op through `start_openpi_shared_ray_runtime`, which places the
worker session inside a Ray actor. The worker session code itself
(`OpenPIPi05WorkerSession`) is pure JAX and knows nothing about Ray; it is already
driven through the in-process `OpenPIDirectWorkerClient` adapter that translates
`request(op, payload)` into direct `session.<op>(payload)` / `_dispatch(...)` calls.

This factory returns that same `OpenPIDirectWorkerClient`, but started in the
current process instead of inside a Ray actor. Injected via
`OpenPIPi05TrainingEngine(runtime_factory=make_local_pi05_runtime)`, it lets the
already-validated LoRA training loop (create_session -> forward_backward ->
optim_step -> save_sampler_weights) run with no Ray involvement at all.

Nothing in the worker, the training engine body, or `OpenPIDirectWorkerClient` is
modified; only the runtime assembly layer (the exact Ray coupling point) is swapped.
"""

from __future__ import annotations

from typing import Any

from mint_server.backend.core.model_registry import ModelConfig
from mint_server.backend.openpi.openpi_direct_runtime import OpenPIDirectWorkerClient
from mint_server.backend.openpi.openpi_fast_runtime import OpenPIFastRuntimeSpec
from mint_server.backend.openpi.openpi_pi05_training import OPENPI_PI05_WORKER_MODULE


async def make_local_pi05_runtime(
    *,
    session: Any,
    model_config: ModelConfig,
    config_name: str,
) -> OpenPIDirectWorkerClient:
    """Start an in-process pi0.5 worker client (no Ray actor).

    Signature matches the `runtime_factory` contract expected by
    `OpenPIPi05TrainingEngine`. `config_name`/`model_config` are accepted for
    parity with the Ray factory; the direct client reads them off the
    create_session payload the engine sends next, so they are unused here.
    """
    del session, model_config, config_name

    # NOTE: deliberately do NOT use OpenPIFastRuntimeSpec.from_env(). That helper
    # runs the PFS runtime-env bootstrap (manifest.json / tier layout) used to
    # build a Ray actor's runtime_env -- pure Ray infra the in-process direct
    # client never touches. The direct client only reads spec.worker_module and
    # the timeout fields, so a plain spec is both sufficient and Ray-free.
    spec = OpenPIFastRuntimeSpec(worker_module=OPENPI_PI05_WORKER_MODULE)
    return await OpenPIDirectWorkerClient.start(spec)


async def make_local_pi05_action_runtime(
    *,
    action_session_id: str,
    base_model: str,
    checkpoint_path: str,
    model_config: ModelConfig,
    config_name: str,
) -> OpenPIDirectWorkerClient:
    """Start an in-process pi0.5 ACTION (inference) worker client (no Ray actor).

    Signature matches the action-session `runtime_factory` contract in
    `action_session_manager._default_pi05_runtime_factory`. The direct client
    reads the checkpoint_path etc. off the create_session payload the manager
    sends next, so those args are accepted only for parity.
    """
    del base_model, checkpoint_path, model_config, config_name

    import os
    from pathlib import Path

    from mint_server.backend.openpi.openpi_pi05_training import (
        OPENPI_PI05_ACTION_WORKER_MODULE,
    )

    # The action worker persists per-session state under this root. The Ray
    # action runtime injected it via actor runtime_env; in-process we set it
    # directly (the direct client runs in this process, so os.environ is visible).
    # Keep it Ray-free: no namespace/actor-name; scope by action_session_id.
    # create_session is serialized (single-tenant A), and the worker reads this
    # env at construction (during the create_session dispatch right after start),
    # so setting it per session here is race-free.
    base_root = str(os.environ.get("MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT_BASE") or "").strip()
    if not base_root:
        mint_code_root = str(os.environ.get("MINT_CODE_ROOT") or "").strip() or "."
        base_root = str(Path(mint_code_root).resolve() / "checkpoints" / "openpi_action_session_state" / "local")
    state_root = Path(base_root) / str(action_session_id)
    state_root.mkdir(parents=True, exist_ok=True)
    os.environ["MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT"] = str(state_root)

    spec = OpenPIFastRuntimeSpec(worker_module=OPENPI_PI05_ACTION_WORKER_MODULE)
    return await OpenPIDirectWorkerClient.start(spec)
