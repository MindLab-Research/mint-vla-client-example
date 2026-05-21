#!/usr/bin/env python
"""Bootstrap Mint detached control-plane actors before API workers start."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_server import (  # noqa: E402
    _load_local_runtime_env_module,
    _reexec_if_env_mismatch,
    _reexec_to_runtime_host_python_if_needed,
    _seed_runtime_env_from_config,
    _set_exact_pythonpath,
    _set_exact_torch_ld_library_path,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to mint-server TOML config file (sets MINT_CONFIG_PATH before import).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Per-step Ray call timeout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable bootstrap summary.",
    )
    return parser.parse_args(argv)


def _prepare_runtime(config_path: str | None) -> None:
    if config_path:
        os.environ["MINT_CONFIG_PATH"] = config_path
        _seed_runtime_env_from_config(config_path)

    _reexec_to_runtime_host_python_if_needed()

    runtime_env = _load_local_runtime_env_module()
    pythonpath = _set_exact_pythonpath(
        runtime_env.bootstrap_runtime_pythonpath(
            os.environ,
            repo_root=str(_REPO_ROOT),
        )
    )
    ld_library_path = _set_exact_torch_ld_library_path()
    _reexec_if_env_mismatch(
        pythonpath=pythonpath,
        ld_library_path=ld_library_path,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _prepare_runtime(str(args.config_path) if args.config_path else None)

    from mint_server.backend import config_actor, model_actor_supervisor
    from mint_server.config import RAY_NAMESPACE
    from mint_server.ray_utils import init_ray

    init_ray(namespace=RAY_NAMESPACE, ignore_reinit_error=True)
    config_snapshot = config_actor.ensure_started(timeout_s=float(args.timeout_s))
    supervisor_snapshot = model_actor_supervisor.ensure_started(timeout_s=float(args.timeout_s))
    summary = {
        "ok": True,
        "config_actor": {
            "actor_name": config_snapshot.get("actor_name"),
            "ray_namespace": config_snapshot.get("ray_namespace"),
            "fingerprint": config_snapshot.get("fingerprint"),
        },
        "model_actor_supervisor": {
            "desired_total": supervisor_snapshot.get("desired_total"),
            "managed_total": supervisor_snapshot.get("managed_total"),
            "reconcile_loop_running": supervisor_snapshot.get("reconcile_loop_running"),
            "last_reconcile_at": supervisor_snapshot.get("last_reconcile_at"),
        },
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            "Mint control plane ready: "
            f"config={summary['config_actor']['actor_name']} "
            f"supervisor_loop={summary['model_actor_supervisor']['reconcile_loop_running']}"
        )


if __name__ == "__main__":
    main()
