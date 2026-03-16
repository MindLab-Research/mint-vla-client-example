import os
import sys
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

RAY_NAMESPACE = os.environ.get("TINKER_RAY_NAMESPACE", "").strip()

PG_BUNDLE_COUNT = os.environ.get("TINKER_REPRO_PG_BUNDLE_COUNT", "").strip()


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(url: str, *, timeout_s: float) -> tuple[int, dict[str, Any] | None, str]:
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    try:
        data = resp.json()
    except Exception:
        data = None
    return resp.status_code, data, resp.text


def main() -> int:
    if not RAY_NAMESPACE:
        return _fail("TINKER_RAY_NAMESPACE is required (run on mint-dev with the server namespace)")
    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    if not ray_address:
        return _fail("RAY_ADDRESS is required and must be the validated GCS address")

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
                return _fail(f"Invalid TINKER_REPRO_PG_BUNDLE_COUNT={PG_BUNDLE_COUNT!r} (expected int)")
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

        code, data, text = _get_json(f"{BASE_URL}/api/v1/healthz", timeout_s=10.0)
        if code == 200:
            return _fail(
                "healthz returned 200 while Ray has pending GPU placement-group demand; "
                f"expected 503. body={text[:400]!r}"
            )
        if code != 503:
            return _fail(f"healthz returned {code}, expected 503. body={text[:400]!r}")
        if not isinstance(data, dict) or data.get("status") != "degraded":
            return _fail(f"healthz 503 returned unexpected json: {data!r}")
        if data.get("reason") != "pending_placement_groups":
            return _fail(f"healthz 503 unexpected reason: {data.get('reason')!r} body={data!r}")

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
