"""Tests for issue #727: dense trainer reaper fix.

Two sub-fixes:
  Fix B  — grace period in ModelActorPlacementReconciler._cleanup_undesired_mint_gpu_actors
  Fix D  — _adopt_surviving_gpu_actors re-registers still-alive workers on supervisor restart

Test matrix
-----------
Grace period (Fix B):
  test_grace_undesired_actor_not_killed_on_first_sight
  test_grace_undesired_actor_not_killed_within_grace
  test_grace_undesired_actor_killed_after_grace_expires
  test_grace_timer_reset_when_actor_becomes_protected
  test_grace_stale_first_seen_purged_when_actor_disappears

Adoption (Fix D):
  test_adopt_surviving_dense_actor_registered
  test_adopt_surviving_dense_actor_marked_ready
  test_adopt_idempotent_already_registered
  test_adopt_wrong_namespace_skipped
  test_adopt_non_mint_actor_skipped
  test_adopt_lister_failure_tolerated
  test_adopt_actor_appears_in_reconcile_protected_names
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mint_server.backend.model_actor_inventory import (
    ActorType,
    ModelActorInventory,
    _ModelActorInventoryState,
)
from mint_server.backend.model_actor_placement import (
    ModelActorPlacementReconciler,
    _undesired_gpu_actor_grace_s,
)
from mint_server.backend.model_actor_supervisor import (
    ModelActorSupervisor,
    ModelActorSupervisorCore,
)


# ---------------------------------------------------------------------------
# Helpers shared by placement tests
# ---------------------------------------------------------------------------

def _make_reconciler(
    namespace: str = "mint",
    killed: list[str] | None = None,
    actors: list[dict[str, Any]] | None = None,
) -> ModelActorPlacementReconciler:
    """Return a reconciler with stub killer and lister."""
    _killed: list[str] = [] if killed is None else killed
    _actors: list[dict[str, Any]] = [] if actors is None else actors

    def _fake_killer(actor_info: dict[str, Any], reason: str) -> bool:
        _killed.append(str(actor_info.get("name") or ""))
        return True

    def _fake_lister():
        return list(_actors)

    r = ModelActorPlacementReconciler(
        namespace=namespace,
        gpu_actor_killer=_fake_killer,
        gpu_actor_lister=_fake_lister,
        placement_group_remover=lambda pg_name, ns: False,
    )
    return r


def _dense_actor(name: str, namespace: str = "mint", gpu: float = 1.0) -> dict[str, Any]:
    return {"name": name, "namespace": namespace, "node_ip": "1.2.3.4", "gpu": gpu}


# ---------------------------------------------------------------------------
# Fix B — grace period
# ---------------------------------------------------------------------------

class TestGracePeriod:
    def test_undesired_actor_not_killed_on_first_sight(self) -> None:
        """First reconcile: actor not in keep set → start grace, do NOT kill."""
        killed: list[str] = []
        actors = [_dense_actor("mint_dense_foo")]
        r = _make_reconciler(killed=killed, actors=actors)

        with patch(
            "mint_server.backend.model_actor_placement._undesired_gpu_actor_grace_s",
            return_value=60.0,
        ):
            cleaned = r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())

        assert cleaned == []
        assert killed == []
        assert "mint_dense_foo" in r._undesired_first_seen

    def test_undesired_actor_not_killed_within_grace(self) -> None:
        """Second reconcile at t+5s: still within grace → do NOT kill."""
        killed: list[str] = []
        actors = [_dense_actor("mint_dense_foo")]
        r = _make_reconciler(killed=killed, actors=actors)

        with patch(
            "mint_server.backend.model_actor_placement._undesired_gpu_actor_grace_s",
            return_value=60.0,
        ):
            r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())
            # Simulate 5 seconds passing (within grace)
            r._undesired_first_seen["mint_dense_foo"] -= 5
            cleaned = r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())

        assert cleaned == []
        assert killed == []
        assert "mint_dense_foo" in r._undesired_first_seen

    def test_undesired_actor_killed_after_grace_expires(self) -> None:
        """After grace_s have passed the actor must be killed."""
        killed: list[str] = []
        actors = [_dense_actor("mint_dense_foo")]
        r = _make_reconciler(killed=killed, actors=actors)

        with patch(
            "mint_server.backend.model_actor_placement._undesired_gpu_actor_grace_s",
            return_value=60.0,
        ):
            r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())
            # Wind clock forward past grace
            r._undesired_first_seen["mint_dense_foo"] -= 61
            cleaned = r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())

        assert "mint_dense_foo" in cleaned
        assert "mint_dense_foo" in killed
        # Entry must be removed after kill
        assert "mint_dense_foo" not in r._undesired_first_seen

    def test_grace_timer_reset_when_actor_becomes_protected(self) -> None:
        """When actor enters keep set, its first_seen entry is cleared."""
        killed: list[str] = []
        actors = [_dense_actor("mint_dense_foo")]
        r = _make_reconciler(killed=killed, actors=actors)

        with patch(
            "mint_server.backend.model_actor_placement._undesired_gpu_actor_grace_s",
            return_value=60.0,
        ):
            # First pass: not in keep set → record grace start
            r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())
            assert "mint_dense_foo" in r._undesired_first_seen

            # Second pass: now protected → clear timer
            r._cleanup_undesired_mint_gpu_actors(
                keep_actor_names={"mint_dense_foo"}
            )

        assert "mint_dense_foo" not in r._undesired_first_seen
        assert killed == []

    def test_stale_first_seen_purged_when_actor_disappears(self) -> None:
        """Entries for actors no longer listed are purged after the loop."""
        killed: list[str] = []
        actors = [_dense_actor("mint_dense_foo")]
        r = _make_reconciler(killed=killed, actors=actors)

        with patch(
            "mint_server.backend.model_actor_placement._undesired_gpu_actor_grace_s",
            return_value=60.0,
        ):
            # Record first_seen for mint_dense_foo
            r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())
            assert "mint_dense_foo" in r._undesired_first_seen

            # Actor disappears from the cluster
            actors.clear()
            r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())

        # Stale entry must be removed
        assert "mint_dense_foo" not in r._undesired_first_seen

    def test_zero_grace_kills_immediately_on_second_call(self) -> None:
        """With grace=0 an actor that appeared on the previous call is killed."""
        killed: list[str] = []
        actors = [_dense_actor("mint_dense_bar")]
        r = _make_reconciler(killed=killed, actors=actors)

        with patch(
            "mint_server.backend.model_actor_placement._undesired_gpu_actor_grace_s",
            return_value=0.0,
        ):
            # First call: records first_seen at now
            r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())
            # Set first_seen slightly in the past so elapsed >= 0
            r._undesired_first_seen["mint_dense_bar"] -= 0.001
            cleaned = r._cleanup_undesired_mint_gpu_actors(keep_actor_names=set())

        assert "mint_dense_bar" in cleaned


# ---------------------------------------------------------------------------
# Fix D — adoption helpers
# ---------------------------------------------------------------------------

def _make_supervisor(monkeypatch: pytest.MonkeyPatch) -> ModelActorSupervisor:
    """Return a fresh in-process ModelActorSupervisor with empty inventory."""
    pool = ModelActorSupervisor()
    import mint_server.backend.model_actor_supervisor as sup_mod

    monkeypatch.setattr(sup_mod, "model_actor_supervisor", pool)
    monkeypatch.setattr(sup_mod, "get_model_actor_supervisor", lambda: pool)
    pool.clear(kill_actors=False)
    return pool


class TestAdoption:
    def test_adopt_surviving_dense_actor_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mint_dense_* actor returned by the lister is added to inventory."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [_dense_actor("mint_dense_alpha", gpu=1.0)]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        entry = pool.get("mint_dense_alpha")
        assert entry is not None
        assert entry.actor_type == ActorType.DENSE
        assert entry.num_gpus == 1
        assert entry.metadata.get("adopted_on_restart") is True

    def test_adopt_surviving_dense_actor_marked_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adopted actor must not be stuck in 'creating' state."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [_dense_actor("mint_dense_beta")]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        entry = pool.get("mint_dense_beta")
        assert entry is not None
        assert entry.creating is False

    def test_adopt_idempotent_already_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling adopt twice does not duplicate the entry."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [_dense_actor("mint_dense_gamma")]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()
            pool._adopt_surviving_gpu_actors()

        # Still only one entry
        entry = pool.get("mint_dense_gamma")
        assert entry is not None
        # No duplicate; inventory count == exactly what we put in
        entries = list(pool.iter_entries())
        names = [e.actor_name for e in entries]
        assert names.count("mint_dense_gamma") == 1

    def test_adopt_wrong_namespace_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Actors in a different namespace must not be adopted."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [
            {"name": "mint_dense_foreign", "namespace": "other", "node_ip": "1.2.3.4", "gpu": 1.0}
        ]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ), patch(
            "mint_server.backend.model_actor_supervisor._ray_namespace",
            return_value="mint",
        ):
            pool._adopt_surviving_gpu_actors()

        assert pool.get("mint_dense_foreign") is None

    def test_adopt_non_mint_actor_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-mint-prefixed actors are not adopted."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [
            {"name": "some_random_actor", "namespace": "mint", "node_ip": "1.2.3.4", "gpu": 1.0}
        ]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        assert pool.get("some_random_actor") is None

    def test_adopt_lister_failure_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the GPU lister raises, adoption logs a warning but does not raise."""
        pool = _make_supervisor(monkeypatch)

        def _boom():
            raise RuntimeError("ray not available")

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            side_effect=_boom,
        ):
            # Must not raise
            pool._adopt_surviving_gpu_actors()

    def test_adopt_actor_appears_in_reconcile_protected_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After adoption, actor name is returned by _reconcile_protected_actor_names."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [_dense_actor("mint_dense_delta")]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        protected = pool._reconcile_protected_actor_names(desired={})
        assert "mint_dense_delta" in protected

    def test_adopt_vllm_actor_gets_correct_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mint_vllm_* actors are adopted with ActorType.VLLM."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [
            {"name": "mint_vllm_qwen3-0-6b", "namespace": "mint", "node_ip": "1.2.3.4", "gpu": 2.0}
        ]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        entry = pool.get("mint_vllm_qwen3-0-6b")
        assert entry is not None
        assert entry.actor_type == ActorType.VLLM
        assert entry.num_gpus == 2

    def test_adopt_megatron_actor_gets_correct_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mint_megatron_* actors are adopted with ActorType.MEGATRON."""
        pool = _make_supervisor(monkeypatch)

        fake_actors = [
            {"name": "mint_megatron_worker0", "namespace": "mint", "node_ip": "1.2.3.4", "gpu": 8.0}
        ]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        entry = pool.get("mint_megatron_worker0")
        assert entry is not None
        assert entry.actor_type == ActorType.MEGATRON
        assert entry.num_gpus == 8

    def test_adopt_skips_model_runtime_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mint_model_runtime_* wrapper actors must NOT be adopted into inventory.

        They pass _is_mint_gpu_actor_name but hit the ``else: continue`` branch
        in type derivation because no specific ActorType is assigned to them.
        """
        pool = _make_supervisor(monkeypatch)

        fake_actors = [
            {
                "name": "mint_model_runtime_training-Qwen-Qwen3-4B-Instruct-2507_replica-0",
                "namespace": "mint",
                "node_ip": "1.2.3.4",
                "gpu": 8.0,
            }
        ]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        # The wrapper actor must NOT appear in the inventory.
        assert pool.get("mint_model_runtime_training-Qwen-Qwen3-4B-Instruct-2507_replica-0") is None
        assert list(pool.iter_entries()) == []


# ---------------------------------------------------------------------------
# Env-var helper
# ---------------------------------------------------------------------------

class TestGraceEnvVar:
    def test_default_grace_is_120(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MINT_UNDESIRED_GPU_ACTOR_GRACE_S", None)
            assert _undesired_gpu_actor_grace_s() == 120.0

    def test_env_var_override(self) -> None:
        with patch.dict("os.environ", {"MINT_UNDESIRED_GPU_ACTOR_GRACE_S": "30"}):
            assert _undesired_gpu_actor_grace_s() == 30.0

    def test_env_var_negative_clamped_to_zero(self) -> None:
        with patch.dict("os.environ", {"MINT_UNDESIRED_GPU_ACTOR_GRACE_S": "-5"}):
            assert _undesired_gpu_actor_grace_s() == 0.0

    def test_env_var_invalid_uses_default(self) -> None:
        with patch.dict("os.environ", {"MINT_UNDESIRED_GPU_ACTOR_GRACE_S": "banana"}):
            assert _undesired_gpu_actor_grace_s() == 120.0


# ---------------------------------------------------------------------------
# End-to-end: adoption + reconciler reaper interaction (Fix D + Fix B)
# ---------------------------------------------------------------------------

class TestAdoptedActorSurvivesReconcilerReaper:
    def test_adopted_actor_survives_reconciler_reaper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adopted dense actor must NOT be killed by the reconciler reaper.

        Scenario (reproduces the original #727 bug):
          1. Supervisor restarts with an empty inventory.
          2. A mint_dense_* actor is still alive on the cluster.
          3. _adopt_surviving_gpu_actors registers it (Fix D).
          4. _reconcile_protected_actor_names with empty desired includes it.
          5. ModelActorPlacementReconciler with grace=0 does NOT kill it.
        """
        # --- Step 1-3: build supervisor, adopt the actor ---
        pool = _make_supervisor(monkeypatch)
        actor_name = "mint_dense_training-Qwen-Qwen3-4B-Instruct-2507_replica-0"
        fake_lister_actors = [_dense_actor(actor_name, namespace="mint", gpu=8.0)]

        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_lister_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        # Confirm adoption
        assert pool.get(actor_name) is not None

        # --- Step 4: protected set includes the adopted actor ---
        protected = pool._reconcile_protected_actor_names(desired={})
        assert actor_name in protected

        # --- Step 5: reconciler with grace=0 does NOT kill the adopted actor ---
        killed: list[str] = []

        def _fake_killer(actor_info: dict[str, Any], reason: str) -> bool:
            killed.append(str(actor_info.get("name") or ""))
            return True

        reconciler = ModelActorPlacementReconciler(
            namespace="mint",
            gpu_actor_killer=_fake_killer,
            gpu_actor_lister=lambda: list(fake_lister_actors),
            placement_group_remover=lambda pg_name, ns: False,
        )

        with patch.dict("os.environ", {"MINT_UNDESIRED_GPU_ACTOR_GRACE_S": "0"}):
            # First call: records first_seen (grace=0, but first call never kills)
            reconciler(desired={}, protected_actor_names=protected)
            # Second call: grace elapsed (wind clock back slightly)
            if actor_name in reconciler._undesired_first_seen:
                reconciler._undesired_first_seen[actor_name] -= 1.0
            reconciler(desired={}, protected_actor_names=protected)

        # The adopted actor must never appear in the kill list.
        assert actor_name not in killed, (
            f"Adopted actor {actor_name!r} was killed by the reconciler reaper — "
            "Fix D protection via _reconcile_protected_actor_names is broken."
        )


class TestReconcileOnceDoesNotReapAdoptedActor:
    """Stronger end-to-end regression: exercises the full
    _reconcile_once_impl → _reconcile_protected_actor_names →
    self._placement_reconciler(..., protected_actor_names=...) wiring.

    The existing test_adopted_actor_survives_reconciler_reaper builds a
    *separate* reconciler and calls it directly, so it would miss a regression
    where _reconcile_once_impl forgets to pass protected_actor_names.  This
    test routes through reconcile_once() with a reconciler injected at
    construction time.
    """

    def test_reconcile_once_does_not_reap_adopted_actor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reconcile_once() with empty desired must NOT kill an adopted dense actor."""
        actor_name = "mint_dense_training-Qwen-Qwen3-4B-Instruct-2507_replica-0"
        fake_lister_actors = [_dense_actor(actor_name, namespace="mint", gpu=8.0)]

        # Recording killer — will be used by the injected reconciler.
        killed: list[str] = []

        def _recording_killer(actor_info: dict[str, Any], reason: str) -> bool:
            killed.append(str(actor_info.get("name") or ""))
            return True

        # Build a real ModelActorPlacementReconciler with the recording killer
        # and the same fake lister, then wrap it so the supervisor can call it
        # as placement_reconciler(desired, protected_actor_names=...).
        reconciler = ModelActorPlacementReconciler(
            namespace="mint",
            gpu_actor_killer=_recording_killer,
            gpu_actor_lister=lambda: list(fake_lister_actors),
            placement_group_remover=lambda pg_name, ns: False,
        )

        # Inject the reconciler at construction so _reconcile_once_impl uses it.
        pool = ModelActorSupervisor(placement_reconciler=reconciler)
        import mint_server.backend.model_actor_supervisor as sup_mod

        monkeypatch.setattr(sup_mod, "model_actor_supervisor", pool)
        monkeypatch.setattr(sup_mod, "get_model_actor_supervisor", lambda: pool)
        pool.clear(kill_actors=False)

        # Adopt the dense actor so it appears in _reconcile_protected_actor_names.
        with patch(
            "mint_server.backend.model_actor_supervisor._default_gpu_actor_lister",
            return_value=fake_lister_actors,
        ):
            pool._adopt_surviving_gpu_actors()

        assert pool.get(actor_name) is not None, "adoption pre-condition failed"
        assert actor_name in pool._reconcile_protected_actor_names(desired={})

        # Run reconcile_once twice with grace=0 so the second call would kill
        # any unprotected actor.  The adopted actor must survive both calls.
        with patch.dict("os.environ", {"MINT_UNDESIRED_GPU_ACTOR_GRACE_S": "0"}):
            asyncio.run(pool.reconcile_once())
            # Wind the grace clock past expiry so the second call would kill if unprotected.
            if actor_name in reconciler._undesired_first_seen:
                reconciler._undesired_first_seen[actor_name] -= 1.0
            asyncio.run(pool.reconcile_once())

        assert actor_name not in killed, (
            f"Adopted actor {actor_name!r} was killed via reconcile_once() — "
            "_reconcile_once_impl is not passing protected_actor_names to the reconciler."
        )
