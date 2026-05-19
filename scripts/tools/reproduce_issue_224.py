import json
import os
import sys
import types


ISSUE_NUMBER = 224


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    print(f"issue={ISSUE_NUMBER}")

    # Enable gateway mode so register_remote_* attempts to persist routing metadata.
    os.environ["MINT_GATEWAY_CONFIG_JSON"] = json.dumps(
        {
            "model_to_upstream": {"Qwen/Qwen3-0.6B": "u1"},
            "upstreams": {"u1": {"base_url": "http://example.invalid:18000", "auth_mode": "pass_through"}},
        }
    )

    import mint_server.gateway as gw
    import mint_server.backend as backend

    gw._gateway_config = None
    gw._remote_sampling_sessions.clear()
    gw._remote_training_models.clear()

    stub_store = types.SimpleNamespace(
        upsert_sampling_session=lambda **_: (_ for _ in ()).throw(RuntimeError("boom_upsert_sampling_session")),
        get_sampling_session=lambda *_: (_ for _ in ()).throw(RuntimeError("boom_get_sampling_session")),
        delete_sampling_session=lambda *_: (_ for _ in ()).throw(RuntimeError("boom_delete_sampling_session")),
        upsert_training_model=lambda **_: (_ for _ in ()).throw(RuntimeError("boom_upsert_training_model")),
        get_training_model=lambda *_: (_ for _ in ()).throw(RuntimeError("boom_get_training_model")),
        delete_training_model=lambda *_: (_ for _ in ()).throw(RuntimeError("boom_delete_training_model")),
    )

    backend.gateway_session_store = stub_store
    sys.modules["mint_server.backend.gateway_session_store"] = stub_store  # Defensive for import paths.

    try:
        gw.register_remote_sampling_session(
            sampling_session_id="sess1",
            upstream_alias="u1",
            base_model="Qwen/Qwen3-0.6B",
        )
    except Exception as e:
        print(f"PASS: register_remote_sampling_session raised: {type(e).__name__}: {e}")
        return 0

    return _fail("BUG: gateway_session_store failure was silently ignored; routing metadata was not persisted")


if __name__ == "__main__":
    raise SystemExit(main())

