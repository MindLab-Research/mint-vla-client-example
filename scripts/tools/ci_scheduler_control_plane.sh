#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYRIGHT_PYTHON_FORCE_VERSION="${PYRIGHT_PYTHON_FORCE_VERSION:-1.1.409}"
export MINT_SCHEDULER_COVERAGE_MIN="${MINT_SCHEDULER_COVERAGE_MIN:-75}"
export MINT_SCHEDULER_COVERAGE_FILE_MIN="${MINT_SCHEDULER_COVERAGE_FILE_MIN:-70}"
export GIT_PAGER=cat
export PAGER=cat
export LESS=FRX
COVERAGE_FILE="${COVERAGE_FILE:-${TMPDIR:-/tmp}/mint_scheduler_control_plane_coverage_data}"
COVERAGE_JSON="${COVERAGE_JSON:-${TMPDIR:-/tmp}/mint_scheduler_control_plane_coverage.json}"

UV_RUN=(uv run)
if [[ "${MINT_CI_UV_NO_SYNC:-}" == "1" ]]; then
  UV_RUN=(uv run --no-sync)
fi

run_step() {
  local name="$1"
  shift

  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::group::${name}"
  else
    printf '\n== %s ==\n' "$name"
  fi

  echo "+ $*"
  "$@"

  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::endgroup::"
  fi
}

PYTHON_TARGETS=(
  mint_server/backend/control_plane_contracts.py
  mint_server/backend/model_work_scheduler.py
  mint_server/backend/model_engine_host.py
  mint_server/backend/model_work_task_gateway.py
  mint_server/backend/model_actor_supervisor.py
  mint_server/backend/engine_liveness.py
  mint_server/backend/cluster_placement_controller.py
  mint_server/backend/model_placement_topology.py
  tests/test_cluster_placement_controller.py
  tests/test_stateless_control_plane_guardrails.py
  tests/test_issue_593_model_work_scheduler.py
  tests/test_issue_593_model_runtime_actor.py
  tests/test_issue_593_model_actor_supervisor.py
  tests/test_model_work_task_gateway.py
  tests/conftest.py
  scripts/tools/verify_scheduler_control_plane.py
)

mapfile -t COMPONENT_TESTS < <(find tests/component/control_plane -name '*.py' -type f | sort)
PY_COMPILE_TARGETS=("${PYTHON_TARGETS[@]}" "${COMPONENT_TESTS[@]}")

COVERAGE_SOURCE_FILES=(
  mint_server/backend/control_plane_contracts.py
  mint_server/backend/model_work_scheduler.py
  mint_server/backend/model_engine_host.py
  mint_server/backend/model_work_task_gateway.py
  mint_server/backend/model_actor_supervisor.py
  mint_server/backend/engine_liveness.py
  mint_server/backend/cluster_placement_controller.py
  mint_server/backend/model_placement_topology.py
)

COVERAGE_FILE_MIN_EXCEPTIONS=(
  mint_server/backend/model_actor_supervisor.py=65
)

COVERAGE_TEST_TARGETS=(
  tests/component/control_plane
  tests/test_cluster_placement_controller.py
  tests/test_stateless_control_plane_guardrails.py
  tests/test_issue_593_model_work_scheduler.py
  tests/test_issue_593_model_runtime_actor.py
  tests/test_issue_593_model_actor_supervisor.py
  tests/test_model_work_task_gateway.py
)

CRITICAL_SUCCESS_PATHS=(
  tests/component/control_plane/test_scheduler_happy.py::test_scheduler_component_happy_path_reaches_retrieve_future
  tests/component/control_plane/test_scheduler_happy.py::test_scheduler_component_manual_assign_happy_path_reaches_retrieve_future
  tests/component/control_plane/test_scheduler_finalize.py::test_scheduler_component_finish_success_commits_terminal_and_releases_projection
  tests/component/control_plane/test_scheduler_runtime.py::test_scheduler_component_runtime_resumes_durable_finalize_after_engine_death
  tests/test_model_work_task_gateway.py::test_scheduler_gateway_retrieve_ready_reads_typed_payload_ref
  tests/test_cluster_placement_controller.py::test_cluster_placement_controller_reserves_capacity_atomically
  tests/test_cluster_placement_controller.py::test_cluster_placement_controller_computes_backend_bundle_requests
)

CRITICAL_FAILURE_PATHS=(
  tests/component/control_plane/test_scheduler_runtime.py::test_scheduler_component_executor_failure_commits_failed_terminal
  tests/component/control_plane/test_scheduler_runtime.py::test_scheduler_component_payload_write_failure_requeues_without_terminal_commit
  tests/component/control_plane/test_scheduler_lifecycle.py::test_scheduler_component_stale_consumer_cannot_finalize_or_fail_active_lease
  tests/component/control_plane/test_scheduler_lifecycle.py::test_scheduler_component_lease_expiry_requeues_for_retry
  tests/component/control_plane/test_scheduler_cancellation.py::test_scheduler_component_cancel_leased_work_removes_scheduler_projection
  tests/component/control_plane/test_scheduler_supervisor.py::test_scheduler_component_placement_pg_blocked_registers_unclaimable_replica
  tests/test_model_work_task_gateway.py::test_scheduler_gateway_submit_admission_reject_is_not_created
  tests/test_cluster_placement_controller.py::test_cluster_placement_controller_times_out_pending_pg_and_blocks_with_backoff
)

run_step "critical success paths" "${UV_RUN[@]}" pytest "${CRITICAL_SUCCESS_PATHS[@]}" -q
run_step "critical failure paths" "${UV_RUN[@]}" pytest "${CRITICAL_FAILURE_PATHS[@]}" -q
run_step "component harness" "${UV_RUN[@]}" pytest tests/component/control_plane -q
run_step "placement controller" "${UV_RUN[@]}" pytest tests/test_cluster_placement_controller.py -q
run_step "stateless guardrails" "${UV_RUN[@]}" pytest tests/test_stateless_control_plane_guardrails.py -q
run_step "issue 593 scheduler" "${UV_RUN[@]}" pytest tests/test_issue_593_model_work_scheduler.py -q
run_step "issue 593 runtime" "${UV_RUN[@]}" pytest tests/test_issue_593_model_runtime_actor.py -q
run_step "issue 593 supervisor" "${UV_RUN[@]}" pytest tests/test_issue_593_model_actor_supervisor.py -q
run_step "contract verifier" "${UV_RUN[@]}" python scripts/tools/verify_scheduler_control_plane.py
run_step "scoped ruff" "${UV_RUN[@]}" ruff check "${PYTHON_TARGETS[@]}" tests/component/control_plane
run_step "scoped pyright" "${UV_RUN[@]}" pyright --project pyrightconfig.scheduler-ci.json
run_step "py_compile" "${UV_RUN[@]}" python -m py_compile "${PY_COMPILE_TARGETS[@]}"
run_step "coverage erase" env COVERAGE_FILE="$COVERAGE_FILE" "${UV_RUN[@]}" coverage erase
run_step "coverage collect" env COVERAGE_FILE="$COVERAGE_FILE" "${UV_RUN[@]}" coverage run \
  --source=mint_server.backend.control_plane_contracts,mint_server.backend.model_work_scheduler,mint_server.backend.model_engine_host,mint_server.backend.model_work_task_gateway,mint_server.backend.model_actor_supervisor,mint_server.backend.engine_liveness,mint_server.backend.cluster_placement_controller,mint_server.backend.model_placement_topology \
  -m pytest "${COVERAGE_TEST_TARGETS[@]}" -q
run_step "coverage hard gate" env COVERAGE_FILE="$COVERAGE_FILE" "${UV_RUN[@]}" coverage report \
  --fail-under="$MINT_SCHEDULER_COVERAGE_MIN" \
  "${COVERAGE_SOURCE_FILES[@]}"
run_step "coverage json" env COVERAGE_FILE="$COVERAGE_FILE" "${UV_RUN[@]}" coverage json \
  -o "$COVERAGE_JSON" \
  "${COVERAGE_SOURCE_FILES[@]}"
run_step "coverage per-file gate" env COVERAGE_JSON="$COVERAGE_JSON" \
  MINT_SCHEDULER_COVERAGE_FILE_MIN="$MINT_SCHEDULER_COVERAGE_FILE_MIN" \
  "${UV_RUN[@]}" python - "${COVERAGE_SOURCE_FILES[@]}" -- "${COVERAGE_FILE_MIN_EXCEPTIONS[@]}" <<'PY'
from __future__ import annotations

import json
import os
import sys

args = sys.argv[1:]
separator = args.index("--")
source_files = args[:separator]
exceptions = dict(item.split("=", 1) for item in args[separator + 1 :])
default_min = float(os.environ["MINT_SCHEDULER_COVERAGE_FILE_MIN"])

with open(os.environ["COVERAGE_JSON"], encoding="utf-8") as handle:
    coverage = json.load(handle)

failures: list[str] = []
for source_file in source_files:
    summary = coverage["files"][source_file]["summary"]
    percent = float(summary["percent_covered"])
    minimum = float(exceptions.get(source_file, default_min))
    print(f"{source_file}: {percent:.2f}% >= {minimum:.2f}%")
    if percent < minimum:
        failures.append(f"{source_file}: {percent:.2f}% < {minimum:.2f}%")

if failures:
    raise SystemExit("coverage per-file gate failed:\n" + "\n".join(failures))
PY
run_step "diff whitespace" git --no-pager diff --check
