"""Tinker-compatible inference server wrapping verl."""

__version__ = "0.1.0"

# Backward-compatibility aliases: expose modules that were moved into
# subpackages at their original top-level import paths.
# New code should use the subpackage paths (e.g. mint_server.ray.ray_utils).
#
# Import order matters: subpackages with no cross-subpackage deps first,
# then dependents.  This avoids circular-import failures in clean
# environments (e.g. CI) where no .pyc cache exists.
import sys as _sys

# 1. leaf subpackages (no eager cross-subpackage imports)
from .ray import ray_utils, runtime_env  # noqa: F401,E402

# 2. checkpoints depends on ray
from .checkpoints import checkpoint_index, checkpoints  # noqa: F401,E402

# 3. config depends on ray + checkpoints
from .config import config_file, config_hydration, runtime_config  # noqa: F401,E402

# 4. observability depends on health (lazy) + routes (lazy) — safe to load now
from .observability import logging_context  # noqa: F401,E402

# 5. gateway depends on observability + ray
from .gateway import gateway, gateway_auth  # noqa: F401,E402

# 6. health depends on observability + ray + config (lazy)
from .health import health_checks, health_state, ray_cluster_health, ray_gcs_metrics  # noqa: F401,E402

# 7. auth depends on config (eager) + gateway (lazy)
from .auth import auth_identity, model_access_control, supported_models_gate  # noqa: F401,E402

# 8. billing depends on config (eager)
from .billing import usage_store  # noqa: F401,E402

# 9. utils depends on observability (eager) + models (external)
from .utils import (  # noqa: F401,E402
    client_compat,
    compatibility,
    download_tokens,
    futures_utils,
    model_input_utils,
    queue_priority,
    sampling_utils,
    webhook,
)

_sys.modules["mint_server.ray_utils"] = ray_utils
_sys.modules["mint_server.runtime_env"] = runtime_env
_sys.modules["mint_server.checkpoint_index"] = checkpoint_index
_sys.modules["mint_server.checkpoints"] = checkpoints
_sys.modules["mint_server.config_file"] = config_file
_sys.modules["mint_server.config_hydration"] = config_hydration
_sys.modules["mint_server.runtime_config"] = runtime_config
_sys.modules["mint_server.gateway"] = gateway
_sys.modules["mint_server.gateway_auth"] = gateway_auth
_sys.modules["mint_server.health_checks"] = health_checks
_sys.modules["mint_server.health_state"] = health_state
_sys.modules["mint_server.ray_cluster_health"] = ray_cluster_health
_sys.modules["mint_server.ray_gcs_metrics"] = ray_gcs_metrics
_sys.modules["mint_server.logging_context"] = logging_context
_sys.modules["mint_server.auth_identity"] = auth_identity
_sys.modules["mint_server.model_access_control"] = model_access_control
_sys.modules["mint_server.supported_models_gate"] = supported_models_gate
_sys.modules["mint_server.usage_store"] = usage_store
_sys.modules["mint_server.client_compat"] = client_compat
_sys.modules["mint_server.compatibility"] = compatibility
_sys.modules["mint_server.download_tokens"] = download_tokens
_sys.modules["mint_server.futures_utils"] = futures_utils
_sys.modules["mint_server.model_input_utils"] = model_input_utils
_sys.modules["mint_server.queue_priority"] = queue_priority
_sys.modules["mint_server.sampling_utils"] = sampling_utils
_sys.modules["mint_server.webhook"] = webhook

del _sys
