#!/usr/bin/env python3
"""Test 3: Concurrent Clients creating vLLM actors."""
import asyncio
import time
import uuid
import httpx

BASE_URL = "http://localhost:8000"
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-0.6B"]


async def poll_future(hc, rid, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        resp = await hc.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": rid},
            headers={"Authorization": "Bearer dummy"},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            await asyncio.sleep(0.5)
        else:
            return {"error": f"HTTP {resp.status_code}"}
    return {"error": "Timeout"}


async def run_client(model_name, client_num):
    info = {"client_num": client_num, "model": model_name, "ok": False, "steps": []}
    hdrs = {"Authorization": "Bearer dummy"}

    async with httpx.AsyncClient() as hc:
        sid = f"conc{client_num}_{uuid.uuid4().hex[:6]}"

        t = time.time()
        resp = await hc.post(
            f"{BASE_URL}/api/v1/create_model",
            json={
                "session_id": sid,
                "model_seq_id": 1,
                "base_model": model_name,
                "lora_config": {"rank": 32},
                "learning_rate": 1e-4,
            },
            headers=hdrs,
            timeout=300,
        )
        r = await poll_future(hc, resp.json().get("request_id"))
        if "error" in r:
            info["err"] = r["error"]
            return info
        mid = r.get("model_id")
        info["steps"].append(("create", time.time() - t))

        tokens = list(range(1, 11))
        datum = {
            "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": tokens, "shape": [10], "dtype": "int64"},
                "loss_mask": {"data": [0.0] * 5 + [1.0] * 5, "shape": [10], "dtype": "float32"},
            },
        }
        t = time.time()
        resp = await hc.post(
            f"{BASE_URL}/api/v1/forward_backward",
            json={"model_id": mid, "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"}},
            headers=hdrs,
            timeout=120,
        )
        await poll_future(hc, resp.json().get("request_id"))
        info["steps"].append(("fb", time.time() - t))

        t = time.time()
        resp = await hc.post(
            f"{BASE_URL}/api/v1/optim_step",
            json={"model_id": mid, "adam_params": {"learning_rate": 1e-4}},
            headers=hdrs,
            timeout=60,
        )
        await poll_future(hc, resp.json().get("request_id"))
        info["steps"].append(("optim", time.time() - t))

        t = time.time()
        resp = await hc.post(
            f"{BASE_URL}/api/v1/save_weights",
            json={"model_id": mid, "name": "test"},
            headers=hdrs,
            timeout=120,
        )
        await poll_future(hc, resp.json().get("request_id"), timeout=300)
        info["steps"].append(("save", time.time() - t))

        info["ok"] = True
    return info


async def main():
    print("TEST 3: CONCURRENT CLIENTS")
    print("=" * 50)

    tasks = [
        run_client(MODELS[0], 1),
        run_client(MODELS[1], 2),
        run_client(MODELS[0], 3),
        run_client(MODELS[1], 4),
    ]

    print(f"Launching {len(tasks)} concurrent clients...")
    t0 = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total = time.time() - t0
    print(f"Completed in {total:.1f}s\n")

    ok_count = 0
    for res in results:
        if isinstance(res, Exception):
            print(f"Exception: {res}")
            continue
        m_short = res["model"].split("/")[-1]
        status = "OK" if res["ok"] else "FAIL"
        print(f"Client {res['client_num']} ({m_short}): {status}")
        if res["ok"]:
            ok_count += 1
        for step_name, step_time in res["steps"]:
            print(f"  {step_name}: {step_time:.1f}s")

    print("\n" + "=" * 50)
    result = "PASS" if ok_count == len(tasks) else "FAIL"
    print(f"RESULT: {result} ({ok_count}/{len(tasks)})")
    return 0 if ok_count == len(tasks) else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
