#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYRIGHT_PYTHON_FORCE_VERSION="${PYRIGHT_PYTHON_FORCE_VERSION:-1.1.409}"

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
  mint_server/backend/model_runtime_actor.py
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
  scripts/tools/verify_scheduler_control_plane.py
)

PYRIGHT_TARGETS=(
  tests/test_cluster_placement_controller.py
  tests/component/control_plane
  tests/test_stateless_control_plane_guardrails.py
  tests/test_model_work_task_gateway.py
  tests/test_issue_593_model_work_scheduler.py
)

mapfile -t COMPONENT_TESTS < <(find tests/component/control_plane -name '*.py' -type f | sort)
PY_COMPILE_TARGETS=("${PYTHON_TARGETS[@]}" "${COMPONENT_TESTS[@]}")

run_step "component harness" uv run pytest tests/component/control_plane -q
run_step "placement controller" uv run pytest tests/test_cluster_placement_controller.py -q
run_step "stateless guardrails" uv run pytest tests/test_stateless_control_plane_guardrails.py -q
run_step "issue 593 scheduler" uv run pytest tests/test_issue_593_model_work_scheduler.py -q
run_step "issue 593 runtime" uv run pytest tests/test_issue_593_model_runtime_actor.py -q
run_step "issue 593 supervisor" uv run pytest tests/test_issue_593_model_actor_supervisor.py -q
run_step "contract verifier" uv run python scripts/tools/verify_scheduler_control_plane.py
run_step "scoped ruff" uv run ruff check "${PYTHON_TARGETS[@]}" tests/component/control_plane
run_step "scoped pyright" uv run pyright "${PYRIGHT_TARGETS[@]}"
run_step "py_compile" uv run python -m py_compile "${PY_COMPILE_TARGETS[@]}"
run_step "diff whitespace" git diff --check
