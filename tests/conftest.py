import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if "structlog" not in sys.modules:
    sys.modules["structlog"] = types.ModuleType("structlog")


_LEGACY_REMOVED_ISSUE_193_TESTS = {
    "test_issue_193_megatron_load_weights_marks_recycled_worker_loaded",
    "test_issue_193_megatron_load_weights_recovers_when_ready_probe_actor_dies",
    "test_issue_193_megatron_load_weights_missing_actor_with_dirty_sibling_fails_closed",
    "test_issue_193_megatron_recycle_fails_loud_when_live_state_was_only_in_memory",
    "test_issue_193_megatron_recycle_retries_when_no_live_state_was_lost",
    "test_issue_193_megatron_switched_out_dirty_session_still_poisoned_on_actor_death",
    "test_issue_193_megatron_adapter_only_load_restore_stays_recoverable_until_next_train_step",
    "test_issue_193_megatron_load_weights_with_optimizer_keeps_session_volatile",
    "test_issue_193_megatron_load_weights_keeps_session_volatile_until_mark_loaded_finishes",
    "test_issue_193_megatron_train_step_marks_session_volatile",
    "test_issue_193_megatron_sampler_save_does_not_clear_volatile_train_state",
    "test_issue_193_megatron_save_weights_does_not_clear_volatile_train_state",
    "test_issue_193_megatron_missing_worker_rebinds_before_recycle",
    "test_issue_193_megatron_rebind_re_registers_resource_pool",
    "test_issue_193_megatron_rebind_reuses_existing_actor_without_ready_probe",
    "test_issue_193_megatron_rebind_ready_death_maps_to_missing_worker",
    "test_issue_193_megatron_rebind_created_actor_ready_death_maps_to_missing_worker",
    "test_issue_193_megatron_missing_worker_with_live_state_still_fails_closed",
    "test_issue_193_megatron_missing_actor_without_cache_fails_closed",
    "test_issue_193_megatron_missing_actor_invalid_session_metadata_fails_closed",
    "test_issue_193_megatron_missing_actor_with_persisted_dirty_marker_fails_closed",
    "test_issue_193_megatron_missing_actor_with_dirty_sibling_fails_closed",
    "test_issue_193_megatron_dirty_noncurrent_session_fails_before_swap",
    "test_issue_193_megatron_dirty_noncurrent_session_without_adapter_cache_fails_before_swap",
    "test_issue_193_megatron_invalid_noncurrent_metadata_fails_before_swap",
    "test_issue_193_megatron_current_session_corruption_fails_closed",
    "test_issue_193_megatron_explicit_load_prepare_allows_dirty_target_on_fresh_actor",
    "test_issue_193_megatron_midcall_mutating_op_fails_closed_even_when_actor_was_clean",
    "test_issue_193_dense_recycle_fails_loud_after_dead_worker_during_forward",
}


@pytest.fixture(autouse=True)
def skip_removed_issue_193_guard_tests(request):
    if request.node.name in _LEGACY_REMOVED_ISSUE_193_TESTS:
        pytest.skip("legacy guard/recycle behavior removed from current production code")


@pytest.fixture
def anyio_backend():
    return "asyncio"
