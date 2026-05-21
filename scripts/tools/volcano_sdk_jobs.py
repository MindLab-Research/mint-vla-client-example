#!/usr/bin/env python3
"""Operate Volcano ML Platform jobs through volcengine-python-sdk.

This is intentionally small and credential-opaque: credentials are resolved by
the SDK default chain on the driver host and are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mint_server.backend.topology import (
    _create_volcano_mlplatform_client,
    _extract_instance_node_ip,
    _job_id,
    _job_name,
    _job_state,
    _object_get,
    _task_gpu_count,
    _volcano_sdk_module,
    build_volcano_create_job_request,
    load_topology_config,
)


def _json_default(value: Any) -> str:
    return str(value)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default))


def _job_summary(job: Any) -> dict[str, Any]:
    resource_config = _object_get(job, "ResourceConfig", "resource_config")
    return {
        "id": _job_id(job),
        "name": _job_name(job),
        "state": _job_state(job),
        "resource_queue_id": _object_get(resource_config, "ResourceQueueId", "resource_queue_id"),
        "gpu_count": _task_gpu_count(job),
    }


def _cmd_list(args: argparse.Namespace) -> None:
    sdk = _volcano_sdk_module()
    client = _create_volcano_mlplatform_client(
        region=args.region,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    page_size = max(1, min(int(args.limit), 100))
    response = client.list_jobs(
        sdk.ListJobsRequest(
            name_contains=args.name_contains,
            page_number=1,
            page_size=page_size,
            state=args.state,
        )
    )
    jobs = [_job_summary(job) for job in (_object_get(response, "Items", "items") or [])]
    _print_json({"jobs": jobs})


def _cmd_instances(args: argparse.Namespace) -> None:
    sdk = _volcano_sdk_module()
    client = _create_volcano_mlplatform_client(
        region=args.region,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    response = client.list_job_instances(
        sdk.ListJobInstancesRequest(job_id=args.job_id, page_number=1, page_size=args.limit)
    )
    instances = []
    for item in _object_get(response, "Items", "items") or []:
        status = _object_get(item, "Status", "status")
        ips = _object_get(item, "Ips", "ips")
        instances.append(
            {
                "id": _object_get(item, "Id", "id"),
                "name": _object_get(item, "Name", "name"),
                "state": str(_object_get(status, "State", "state") or "").strip(),
                "primary_ip": _object_get(ips, "PrimaryIp", "primary_ip"),
                "host_ip": _object_get(ips, "HostIp", "host_ip"),
                "node_ip": _extract_instance_node_ip(item),
            }
        )
    _print_json({"job_id": args.job_id, "instances": instances})


def _cmd_submit_topology_node(args: argparse.Namespace) -> None:
    config = load_topology_config(args.config)
    node = config.nodes[args.alias]
    request = build_volcano_create_job_request(config, node)
    client = _create_volcano_mlplatform_client(
        region=args.region,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    response = client.create_job(request)
    _print_json(
        {
            "submitted": True,
            "alias": args.alias,
            "job_name": request.name,
            "job_id": _object_get(response, "Id", "id", "JobId", "job_id"),
        }
    )


def _cmd_stop(args: argparse.Namespace) -> None:
    sdk = _volcano_sdk_module()
    client = _create_volcano_mlplatform_client(
        region=args.region,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    response = client.stop_job(
        sdk.StopJobRequest(id=args.job_id, reason=args.reason, dry_run=args.dry_run)
    )
    _print_json({"stopped": not args.dry_run, "dry_run": bool(args.dry_run), "job_id": args.job_id, "response": response})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="cn-beijing")
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=10.0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="List jobs")
    list_p.add_argument("--name-contains", default="")
    list_p.add_argument("--state", default=None)
    list_p.add_argument("--limit", type=int, default=100)
    list_p.set_defaults(func=_cmd_list)

    inst_p = sub.add_parser("instances", help="List job instances and IPs")
    inst_p.add_argument("--job-id", required=True)
    inst_p.add_argument("--limit", type=int, default=20)
    inst_p.set_defaults(func=_cmd_instances)

    submit_p = sub.add_parser("submit-topology-node", help="Submit one topology node by alias")
    submit_p.add_argument("--config", required=True)
    submit_p.add_argument("--alias", required=True)
    submit_p.set_defaults(func=_cmd_submit_topology_node)

    stop_p = sub.add_parser("stop", help="Stop one job")
    stop_p.add_argument("--job-id", required=True)
    stop_p.add_argument("--reason", default="mint operator stop")
    stop_p.add_argument("--dry-run", action="store_true")
    stop_p.set_defaults(func=_cmd_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
