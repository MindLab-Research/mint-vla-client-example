"""Module-level worker function for spawn test."""
import os
import sys

RESULT_FILE = "/vePFS-Mindverse/share/code/spawn_worker_result.txt"

def spawn_worker_check():
    """Worker function that checks vLLM import location."""
    try:
        import vllm
        import importlib
        try:
            import sitecustomize as sc  # type: ignore[import-not-found]
        except Exception:
            sc = None

        with open(RESULT_FILE, "w") as f:
            f.write(f"vllm.__file__: {vllm.__file__}\n")
            f.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')[:500]}\n")
            f.write(f"sys.path[:10]: {sys.path[:10]}\n")
            f.write(f"has_site: {'site' in sys.modules}\n")
            f.write(f"has_sitecustomize: {'sitecustomize' in sys.modules}\n")
            f.write(f"sitecustomize.__file__: {getattr(sc, '__file__', None)}\n")

            try:
                lw = importlib.import_module("vllm.lora.lora_weights")
                Packed = getattr(lw, "PackedLoRALayerWeights", None)
                pack = getattr(Packed, "pack_moe", None) if Packed is not None else None
                f.write(f"pack_moe: {pack}\n")
                f.write(
                    f"pack_moe.__mint_sparse_ok__: {getattr(pack, '__mint_sparse_ok__', None)}\n"
                )
            except Exception as e:
                f.write(f"pack_moe_probe_error: {type(e).__name__}: {e}\n")
    except Exception as e:
        import traceback
        with open(RESULT_FILE, "w") as f:
            f.write(f"spawn_worker_check_failed: {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
