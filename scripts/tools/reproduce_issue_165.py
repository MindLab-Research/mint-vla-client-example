import os
import sys
import time
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL")
if not BASE_URL:
    port = os.environ.get("MINT_PORT", "8000")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("MINT_API_KEY", "dummy")

MODEL = os.environ.get("MINT_MODEL", "Qwen/Qwen3-0.6B")
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
LR = float(os.environ.get("MINT_LR", "5e-5"))

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "900"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "1.0"))

PREWARM_WAIT_S = float(os.environ.get("MINT_PREWARM_WAIT_S", "600"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, body: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, headers=_headers(), json=body, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]!r}")
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(out)}")
    return out


def _get_json(path: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, headers=_headers(), timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:500]!r}")
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"GET {path} returned non-dict json: {type(out)}")
    return out


def _poll_future(request_id: str) -> dict[str, Any]:
    t0 = time.time()
    while True:
        r = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if r.status_code == 408:
            if time.time() - t0 > POLL_TIMEOUT_S:
                raise RuntimeError(f"retrieve_future timeout after {POLL_TIMEOUT_S:.1f}s request_id={request_id}")
            time.sleep(POLL_SLEEP_S)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {r.status_code}: {r.text[:500]!r}")
        out = r.json()
        if not isinstance(out, dict):
            raise RuntimeError(f"retrieve_future returned non-dict json: {type(out)}")
        return out


def _matches_model(entry_base_model: str, model: str) -> bool:
    if entry_base_model == model:
        return True
    needle = model.replace("/", "--").lower()
    return needle in entry_base_model.lower()


def _list_actors() -> list[dict[str, Any]]:
    out = _get_json("/internal/actors", timeout_s=30.0)
    actors = out.get("actors", [])
    if not isinstance(actors, list):
        raise RuntimeError(f"actors field missing/invalid: {actors!r}")
    return [a for a in actors if isinstance(a, dict)]


def _find_training_actors() -> list[dict[str, Any]]:
    out = []
    for a in _list_actors():
        t = a.get("actor_type")
        bm = a.get("base_model")
        if t not in ("dense", "megatron"):
            continue
        if isinstance(bm, str) and bm:
            if not _matches_model(bm, MODEL):
                continue
        out.append(a)
    return out


def _kill_training_actors(actor_type: str) -> None:
    _post_json("/internal/actors/kill", {"actor_type": actor_type}, timeout_s=30.0)


def _create_training_model() -> str:
    session_id = _post_json("/api/v1/create_session", {"tags": ["issue-165"]}, timeout_s=30.0).get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session returned invalid session_id={session_id!r}")
    req = _post_json(
        "/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": MODEL,
            "lora_config": {"rank": LORA_RANK},
            "learning_rate": LR,
        },
        timeout_s=60.0,
    )
    request_id = req.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"create_model returned invalid request_id={request_id!r}")
    out = _poll_future(request_id)
    if "error" in out:
        raise RuntimeError(f"create_model failed: {out.get('error')!r}")
    model_id = out.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model returned invalid model_id={model_id!r}")
    return model_id


def _wait_for_prewarm_training_actor() -> dict[str, Any]:
    t0 = time.time()
    while True:
        acts = _find_training_actors()
        if acts:
            # Prefer matching base_model when available, then prefer ready actors.
            def _score(a: dict[str, Any]) -> tuple[int, int, str]:
                bm = a.get("base_model")
                matches = 0
                if isinstance(bm, str) and bm:
                    matches = 1 if _matches_model(bm, MODEL) else 0
                return (0 if matches else 1, 1 if bool(a.get("creating")) else 0, str(a.get("actor_name") or ""))

            acts.sort(key=_score)
            a = acts[0]
            actor_name = a.get("actor_name")
            actor_type = a.get("actor_type")
            if isinstance(actor_name, str) and actor_name and isinstance(actor_type, str) and actor_type:
                return a
        if time.time() - t0 > PREWARM_WAIT_S:
            raise RuntimeError(f"timed out waiting for prewarm training actor for {MODEL!r} after {PREWARM_WAIT_S:.1f}s")
        time.sleep(1.0)


def main() -> int:
    try:
        _get_json("/api/v1/healthz", timeout_s=5.0)

        # Wait for prewarm to create and protect a training actor (baseline persistent semantics).
        pre = _wait_for_prewarm_training_actor()
        prewarm_actor_name = pre.get("actor_name")
        actor_type = pre.get("actor_type")
        prewarm_protected = bool(pre.get("protected"))
        if not isinstance(prewarm_actor_name, str) or not prewarm_actor_name or not isinstance(actor_type, str) or not actor_type:
            return _fail(f"invalid prewarm actor fields: {pre!r}")
        if not prewarm_protected:
            return _fail(
                f"prewarm training actor is not protected (server not started with MINT_PERSISTENT_MODELS?): "
                f"actor_type={actor_type} actor_name={prewarm_actor_name}"
            )
        print(f"prewarm actor_type={actor_type} actor_name={prewarm_actor_name} protected=1")

        # Kill training actor(s) and ensure on-demand recreate stays protected.
        _kill_training_actors(actor_type)

        t0 = time.time()
        while True:
            acts = _find_training_actors()
            if not acts:
                break
            if time.time() - t0 > 60:
                raise RuntimeError(f"timed out waiting for training actors to disappear after kill: {acts[:2]!r}")
            time.sleep(1.0)

        _create_training_model()

        t0 = time.time()
        post: dict[str, Any] | None = None
        while True:
            acts = _find_training_actors()
            if acts:
                acts.sort(key=lambda a: (bool(a.get("creating")), a.get("actor_name", "")))
                post = acts[0]
                break
            if time.time() - t0 > PREWARM_WAIT_S:
                raise RuntimeError(f"timed out waiting for recreated training actor for {MODEL!r}")
            time.sleep(1.0)

        if not isinstance(post, dict):
            return _fail(f"recreated actor missing/invalid: {post!r}")
        new_actor_name = post.get("actor_name")
        new_protected = bool(post.get("protected"))
        if not isinstance(new_actor_name, str) or not new_actor_name:
            return _fail(f"recreated actor_name invalid: {post!r}")
        if not new_protected:
            return _fail(f"recreated training actor is not protected: actor_type={actor_type} actor_name={new_actor_name}")

        print(f"recreated actor_type={actor_type} actor_name={new_actor_name} protected=1")
        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
