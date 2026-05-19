import asyncio
import time

from mint_server.backend.lora_registry import LoRARegistry


def test_lru_candidates_skip_recently_used(monkeypatch):
    monkeypatch.setenv("MINT_LORA_EVICT_MIN_IDLE_S", "5.0")

    reg = LoRARegistry()

    async def run():
        active = await reg.allocate("active")
        idle = await reg.allocate("idle")

        now = time.time()
        reg._slot_info[active].last_used = now
        reg._slot_info[idle].last_used = now - 60.0

        cands = await reg.get_lru_candidates(1)
        assert cands == [idle]

    asyncio.run(run())


def test_lru_candidates_invalid_env_var_falls_back(monkeypatch):
    monkeypatch.setenv("MINT_LORA_EVICT_MIN_IDLE_S", "not-a-float")

    reg = LoRARegistry()

    async def run():
        active = await reg.allocate("active")
        idle = await reg.allocate("idle")

        now = time.time()
        reg._slot_info[active].last_used = now
        reg._slot_info[idle].last_used = now - 60.0

        cands = await reg.get_lru_candidates(1)
        assert cands == [idle]

    asyncio.run(run())

