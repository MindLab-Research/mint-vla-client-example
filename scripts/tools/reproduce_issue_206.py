import os
import sys
import asyncio
import uuid


RAY_NAMESPACE = os.environ.get("MINT_RAY_NAMESPACE", "").strip()

PG_BUNDLE_COUNT = os.environ.get("MINT_REPRO_PG_BUNDLE_COUNT", "").strip()


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    if not RAY_NAMESPACE:
        return _fail("MINT_RAY_NAMESPACE is required (run on mint-dev with the server namespace)")
    ray_address = os.environ.get("MINT_RAY_GCS_ADDRESS", "").strip()
    if not ray_address:
        return _fail("MINT_RAY_GCS_ADDRESS is required and must be the validated GCS address")

    try:
        import ray
    except Exception as e:
        return _fail(f"ray import failed (run this repro on mint-dev): {type(e).__name__}: {e}")

    ray.init(address=ray_address, namespace=RAY_NAMESPACE, ignore_reinit_error=True)

    pg = None
    pg_name = f"repro_206_pg_{uuid.uuid4().hex[:8]}"

    try:
        cr0 = ray.cluster_resources()
        gpu_total = float(cr0.get("GPU", 0) or 0)
        if gpu_total <= 0:
            return _fail(f"Ray cluster has no GPUs: GPU_total={gpu_total}")

        if PG_BUNDLE_COUNT:
            try:
                pg_bundles = int(PG_BUNDLE_COUNT)
            except Exception:
                return _fail(f"Invalid MINT_REPRO_PG_BUNDLE_COUNT={PG_BUNDLE_COUNT!r} (expected int)")
        else:
            # Force a pending placement group by requesting strictly more bundles than the cluster can satisfy.
            pg_bundles = int(gpu_total) + 1

        if pg_bundles <= 0:
            return _fail(f"Computed pg_bundles={pg_bundles} (GPU_total={gpu_total})")

        bundles = [{"CPU": 1.0, "GPU": 1.0} for _ in range(pg_bundles)]
        pg = ray.util.placement_group(
            bundles,
            strategy="PACK",
            name=pg_name,
            lifetime="detached",
        )

        try:
            ray.get(pg.ready(), timeout=5)
            info = ray.util.placement_group_table(pg)
            return _fail(f"Expected placement group to remain pending, but it became ready: state={info.get('state')}")
        except ray.exceptions.GetTimeoutError:
            pass

        info = ray.util.placement_group_table(pg)
        state = info.get("state")
        if state in ("CREATED", "REMOVED"):
            return _fail(f"Expected placement group to be pending, got state={state!r}")

        from mint_server.backend.ray_cluster.async_ray_control import async_pending_gpu_pg_observation

        obs = asyncio.run(async_pending_gpu_pg_observation(timeout_s=10.0))
        if not isinstance(obs, dict):
            return _fail(f"pending placement-group observation missing: {obs!r}")
        if obs.get("reason") != "pending_placement_groups":
            return _fail(f"pending placement-group observation unexpected reason: {obs.get('reason')!r} body={obs!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")
    finally:
        try:
            if pg is not None:
                ray.util.remove_placement_group(pg)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
