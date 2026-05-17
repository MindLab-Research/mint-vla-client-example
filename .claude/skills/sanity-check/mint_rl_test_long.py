import argparse
import atexit
import json
import os
import random
import re
import unicodedata
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None
import mint
import math
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
from dotenv import load_dotenv
from mint import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.retry_handler import RetryConfig

DEFAULT_TIMEOUT_S = float(os.environ.get("MINT_TEST_TIMEOUT_S", "10800"))
_LAST_REQUEST_ID_BY_TYPE: dict[str, str] = {}
_TIMING_EVENTS: list[dict[str, Any]] = []
_TIMING_JSONL_PATH: Path | None = None
_TIMING_SUMMARY_JSON_PATH: Path | None = None
_TIMING_SUMMARY_MD_PATH: Path | None = None
_TIMING_REPORT_WRITTEN = False
_EXPERIMENT_WALL_START = time.time()


def _dbg(msg: str) -> None:
    print(f"[debug] {msg}", flush=True)

def _ts_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _default_stage_name(label: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", label)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace(" ", "_").replace("-", "_").lower()


def _append_timing_event(
    stage: str,
    *,
    elapsed_s: float,
    status: str,
    step_idx: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    rec: dict[str, Any] = {
        "ts_utc": _ts_utc(),
        "stage": stage,
        "elapsed_s": round(float(elapsed_s), 3),
        "status": status,
    }
    base_model = globals().get("BASE_MODEL")
    if base_model:
        rec["base_model"] = base_model
    if step_idx is not None:
        rec["step"] = int(step_idx) + 1
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            rec[key] = value if isinstance(value, (str, int, float, bool, list, dict)) else str(value)
    _TIMING_EVENTS.append(rec)
    if _TIMING_JSONL_PATH is None:
        return
    with open(_TIMING_JSONL_PATH, "a", encoding="utf-8") as f:
        json.dump(rec, f, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def _timing_summary() -> dict[str, Any]:
    per_stage: dict[str, list[dict[str, Any]]] = {}
    for rec in _TIMING_EVENTS:
        per_stage.setdefault(str(rec["stage"]), []).append(rec)

    stage_summaries: list[dict[str, Any]] = []
    for stage, recs in per_stage.items():
        elapsed = sorted(float(rec["elapsed_s"]) for rec in recs)
        total_s = sum(elapsed)
        count = len(elapsed)
        ok_count = sum(1 for rec in recs if rec.get("status") == "ok")
        error_count = sum(1 for rec in recs if rec.get("status") == "error")
        stage_summaries.append(
            {
                "stage": stage,
                "count": count,
                "ok_count": ok_count,
                "error_count": error_count,
                "total_s": round(total_s, 3),
                "avg_s": round(total_s / count, 3),
                "p50_s": round(elapsed[count // 2], 3),
                "max_s": round(elapsed[-1], 3),
            }
        )

    stage_summaries.sort(key=lambda item: (-float(item["total_s"]), str(item["stage"])))
    return {
        "base_model": globals().get("BASE_MODEL"),
        "event_count": len(_TIMING_EVENTS),
        "wall_clock_s": round(time.time() - _EXPERIMENT_WALL_START, 3),
        "stages": stage_summaries,
    }


def _write_timing_reports() -> None:
    global _TIMING_REPORT_WRITTEN
    if _TIMING_REPORT_WRITTEN or _TIMING_SUMMARY_JSON_PATH is None or _TIMING_SUMMARY_MD_PATH is None:
        return

    summary = _timing_summary()
    with open(_TIMING_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
        f.write("\n")

    md_lines = [
        "# Timing Report",
        "",
        f"- base_model: `{summary.get('base_model')}`",
        f"- wall_clock_s: `{summary['wall_clock_s']}`",
        f"- event_count: `{summary['event_count']}`",
        "",
        "| stage | count | ok | error | total_s | avg_s | p50_s | max_s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["stages"]:
        md_lines.append(
            f"| {item['stage']} | {item['count']} | {item['ok_count']} | {item['error_count']} | "
            f"{item['total_s']:.3f} | {item['avg_s']:.3f} | {item['p50_s']:.3f} | {item['max_s']:.3f} |"
        )
    with open(_TIMING_SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("Timing summary:")
    for item in summary["stages"]:
        print(
            f"[timing] stage={item['stage']} count={item['count']} ok={item['ok_count']} "
            f"error={item['error_count']} total_s={item['total_s']:.3f} "
            f"avg_s={item['avg_s']:.3f} p50_s={item['p50_s']:.3f} max_s={item['max_s']:.3f}"
        )
    print(f"[timing] wall_clock_s={summary['wall_clock_s']:.3f} event_count={summary['event_count']}")
    print(f"[timing] timing_events_jsonl={_TIMING_JSONL_PATH}")
    print(f"[timing] timing_summary_json={_TIMING_SUMMARY_JSON_PATH}")
    print(f"[timing] timing_summary_md={_TIMING_SUMMARY_MD_PATH}")
    _TIMING_REPORT_WRITTEN = True


def _time_call(
    fn: Any,
    *,
    stage_name: str,
    step_idx: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    start = time.time()
    try:
        result = fn()
    except Exception as e:
        event_extra = dict(extra or {})
        event_extra["error_type"] = type(e).__name__
        _append_timing_event(
            stage_name,
            elapsed_s=time.time() - start,
            status="error",
            step_idx=step_idx,
            extra=event_extra,
        )
        raise
    _append_timing_event(
        stage_name,
        elapsed_s=time.time() - start,
        status="ok",
        step_idx=step_idx,
        extra=extra,
    )
    return result


def _result_with_heartbeat(
    fut: Any,
    *,
    label: str,
    timeout_s: float,
    stage_name: str | None = None,
    step_idx: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    # Prefer polling an underlying concurrent Future so we can print heartbeats
    # even if the public `.result(timeout=...)` doesn't respect timeouts.
    cfut = None
    try:
        if hasattr(fut, "future") and callable(getattr(fut, "future")):
            cfut = fut.future()
    except Exception:
        cfut = None
    if cfut is None:
        try:
            inner = getattr(fut, "_future", None)
            if inner is not None and hasattr(inner, "future") and callable(getattr(inner, "future")):
                cfut = inner.future()
        except Exception:
            cfut = None

    start = time.time()
    last_print = 0.0
    stage = stage_name or _default_stage_name(label)
    try:
        while True:
            elapsed = time.time() - start
            if timeout_s > 0 and elapsed >= timeout_s:
                result = fut.result(timeout=0.01)
                break
            if cfut is not None:
                if cfut.done():
                    result = cfut.result()
                    break
            else:
                try:
                    result = fut.result(timeout=0.2)
                    break
                except TimeoutError:
                    pass
            if elapsed - last_print >= 10.0:
                rid = _LAST_REQUEST_ID_BY_TYPE.get(label)
                if rid:
                    _dbg(f"waiting label={label} request_id={rid} elapsed_s={elapsed:.1f}")
                else:
                    _dbg(f"waiting label={label} elapsed_s={elapsed:.1f}")
                last_print = elapsed
            time.sleep(0.2)
    except Exception as e:
        event_extra = dict(extra or {})
        event_extra["label"] = label
        event_extra["request_id"] = _LAST_REQUEST_ID_BY_TYPE.get(label)
        event_extra["error_type"] = type(e).__name__
        _append_timing_event(
            stage,
            elapsed_s=time.time() - start,
            status="error",
            step_idx=step_idx,
            extra=event_extra,
        )
        raise

    event_extra = dict(extra or {})
    event_extra["label"] = label
    event_extra["request_id"] = _LAST_REQUEST_ID_BY_TYPE.get(label)
    _append_timing_event(
        stage,
        elapsed_s=time.time() - start,
        status="ok",
        step_idx=step_idx,
        extra=event_extra,
    )
    return result


def _install_tinker_future_debug() -> None:
    """
    Print server-side request_id for each internal Tinker APIFuture created.

    This catches request_ids for create_model, asample, retrieve_future loops, etc,
    even when public mint/tinker APIs only expose blocking `.result()`.
    """
    try:
        from tinker.lib.api_future_impl import _APIFuture as _InternalAPIFuture  # type: ignore
    except Exception as e:
        _dbg(f"cannot import tinker.lib.api_future_impl._APIFuture: {type(e).__name__}: {e}")
        return

    if getattr(_InternalAPIFuture, "_mint_debug_patched", False):
        return

    real_init = _InternalAPIFuture.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Signature: (model_cls, holder, untyped_future, request_start_time, request_type, queue_state_observer=None)
        untyped_future = None
        request_type = None
        try:
            if len(args) >= 3:
                untyped_future = args[2]
            if len(args) >= 5:
                request_type = args[4]
            if request_type is None:
                request_type = kwargs.get("request_type")
            if untyped_future is None:
                untyped_future = kwargs.get("untyped_future")
        except Exception:
            pass

        real_init(self, *args, **kwargs)

        try:
            rid = getattr(untyped_future, "request_id", None)
            if rid:
                if request_type:
                    _LAST_REQUEST_ID_BY_TYPE[str(request_type)] = str(rid)
                _dbg(f"api_future created request_type={request_type} request_id={rid}")
        except Exception:
            pass

    _InternalAPIFuture.__init__ = patched_init  # type: ignore[method-assign]
    _InternalAPIFuture._mint_debug_patched = True  # type: ignore[attr-defined]
    _dbg("installed tinker APIFuture request_id logger")


def _patch_trust_remote_code_for_tokenizers(*, enable: bool) -> None:
    """
    Ensure non-interactive runs don't hang on tokenizer trust_remote_code prompts.

    This only affects the local client process that runs this script.
    """
    if not enable:
        return

    # Patch transformers AutoTokenizer directly (this is what tinker uses internally).
    try:
        from transformers import AutoTokenizer as _HF_AutoTokenizer  # type: ignore
    except Exception as e:
        _dbg(f"cannot import transformers.AutoTokenizer: {type(e).__name__}: {e}")
        _HF_AutoTokenizer = None

    if _HF_AutoTokenizer is not None and not getattr(
        _HF_AutoTokenizer, "_mint_trust_remote_code_patched", False
    ):
        cm = _HF_AutoTokenizer.__dict__.get("from_pretrained")
        real_fn = getattr(cm, "__func__", None)
        if real_fn is None:
            # Fallback: already-resolved descriptor (should still work as a function).
            real_fn = _HF_AutoTokenizer.from_pretrained  # type: ignore[assignment]

        def wrapped_from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            # Moonlight tokenizer requires custom code; always allow remote code when this
            # script is explicitly running Moonlight.
            kwargs.setdefault("trust_remote_code", True)
            return real_fn(cls, pretrained_model_name_or_path, *args, **kwargs)

        _HF_AutoTokenizer.from_pretrained = classmethod(wrapped_from_pretrained)  # type: ignore[method-assign]
        _HF_AutoTokenizer._mint_trust_remote_code_patched = True  # type: ignore[attr-defined]
        _dbg("patched transformers.AutoTokenizer.from_pretrained trust_remote_code for Moonlight tokenizer")

    # Best-effort: patch any re-exported AutoTokenizer under tinker too (older SDK variants).
    try:
        import tinker.lib.public_interfaces.sampling_client as sampling_client  # type: ignore
    except Exception:
        return

    if getattr(sampling_client, "_mint_trust_remote_code_patched", False):
        return
    AutoTokenizer2 = getattr(sampling_client, "AutoTokenizer", None)
    if AutoTokenizer2 is None:
        return
    if getattr(AutoTokenizer2, "_mint_trust_remote_code_patched", False):
        sampling_client._mint_trust_remote_code_patched = True  # type: ignore[attr-defined]
        return
    cm2 = AutoTokenizer2.__dict__.get("from_pretrained")
    real2 = getattr(cm2, "__func__", None)
    if real2 is None:
        real2 = AutoTokenizer2.from_pretrained  # type: ignore[assignment]

    def wrapped2(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        kwargs.setdefault("trust_remote_code", True)
        return real2(cls, pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer2.from_pretrained = classmethod(wrapped2)  # type: ignore[method-assign]
    AutoTokenizer2._mint_trust_remote_code_patched = True  # type: ignore[attr-defined]
    sampling_client._mint_trust_remote_code_patched = True  # type: ignore[attr-defined]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MinT RL arithmetic demo (long prompts)")
    parser.add_argument(
        "--timeout-s",
        dest="timeout_s",
        type=float,
        default=float(os.environ.get("MINT_TEST_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))),
        help="Per-request timeout seconds (default: env MINT_TEST_TIMEOUT_S or 10800; use <=0 to disable).",
    )
    parser.add_argument(
        "--model",
        "--base-model",
        dest="base_model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name (as listed by server capabilities)",
    )
    parser.add_argument(
        "--lora-rank",
        dest="lora_rank",
        type=int,
        default=int(os.environ.get("MINT_TEST_LORA_RANK", "0")),
        help=(
            "LoRA rank (default: 0=auto; env: MINT_TEST_LORA_RANK). "
            "For K2, pick a value divisible by vLLM TP."
        ),
    )
    parser.add_argument(
        "--inference-only",
        dest="inference_only",
        action="store_true",
        default=False,
        help="Skip LoRA training and only run sampling on base_model (default: disabled).",
    )
    parser.add_argument(
        "--force-train",
        dest="force_train",
        action="store_true",
        default=False,
        help="Allow training path even for models defaulted to inference-only (e.g. routed models).",
    )
    parser.add_argument(
        "--train-mlp",
        dest="train_mlp",
        action="store_true",
        default=None,
        help="Train MLP LoRA (default: model-specific).",
    )
    parser.add_argument(
        "--no-train-mlp",
        dest="train_mlp",
        action="store_false",
        help="Disable MLP LoRA training.",
    )
    parser.add_argument(
        "--train-attn",
        dest="train_attn",
        action="store_true",
        default=None,
        help="Train attention LoRA (default: enabled).",
    )
    parser.add_argument(
        "--no-train-attn",
        dest="train_attn",
        action="store_false",
        help="Disable attention LoRA training.",
    )
    parser.add_argument(
        "--train-unembed",
        dest="train_unembed",
        action="store_true",
        default=None,
        help="Train unembed LoRA (default: enabled).",
    )
    parser.add_argument(
        "--no-train-unembed",
        dest="train_unembed",
        action="store_false",
        help="Disable unembed LoRA training.",
    )
    parser.add_argument(
        "--system-prompt-path",
        dest="system_prompt_path",
        default="./prompts/claude_system.txt",
        help="Path to the system prompt to prepend to each input",
    )
    parser.add_argument(
        "--print-token-counts",
        dest="print_token_counts",
        action="store_true",
        default=True,
        help="Print token counts for inputs and outputs (default: enabled)",
    )
    parser.add_argument(
        "--no-print-token-counts",
        dest="print_token_counts",
        action="store_false",
        help="Disable token-count printing",
    )
    parser.add_argument(
        "--save-ckpt",
        dest="save_ckpt",
        action="store_true",
        default=True,
        help="Save checkpoints at the end (default: enabled)",
    )
    parser.add_argument(
        "--no-save-ckpt",
        dest="save_ckpt",
        action="store_false",
        help="Disable end-of-run checkpoint save",
    )
    parser.add_argument(
        "--print-sample-text",
        dest="print_sample_text",
        action="store_true",
        default=False,
        help="Print decoded text for each sampled sequence (default: disabled).",
    )
    parser.add_argument(
        "--num-rl-steps",
        dest="num_rl_steps",
        type=int,
        default=int(os.environ.get("MINT_TEST_NUM_RL_STEPS", "5")),
        help="Number of RL steps (default: 5; env: MINT_TEST_NUM_RL_STEPS)",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=int(os.environ.get("MINT_TEST_BATCH_SIZE", "8")),
        help="Problems per RL step (default: 8; env: MINT_TEST_BATCH_SIZE)",
    )
    parser.add_argument(
        "--group-size",
        dest="group_size",
        type=int,
        default=int(os.environ.get("MINT_TEST_GROUP_SIZE", "8")),
        help="Samples per problem (default: 8; env: MINT_TEST_GROUP_SIZE)",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=int(os.environ.get("MINT_TEST_MAX_TOKENS", "4096")),
        help="Max generation tokens per sample (default: 4096; env: MINT_TEST_MAX_TOKENS)",
    )
    parser.add_argument(
        "--prompt-min-tokens",
        dest="prompt_min_tokens",
        type=int,
        default=int(os.environ.get("MINT_TEST_PROMPT_MIN_TOKENS", "0")),
        help="Ensure each prompt has at least this many input tokens by prefix-padding (default: 0).",
    )
    parser.add_argument(
        "--prompt-target-tokens",
        dest="prompt_target_tokens",
        type=int,
        default=int(os.environ.get("MINT_TEST_PROMPT_TARGET_TOKENS", "0")),
        help=(
            "Ensure each prompt has exactly this many input tokens via token-level prefix padding "
            "(default: 0). If set, this takes precedence over --prompt-min-tokens."
        ),
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=float(os.environ.get("MINT_TEST_TEMPERATURE", "0.7")),
        help="Sampling temperature (default: 0.7; env: MINT_TEST_TEMPERATURE)",
    )
    parser.add_argument(
        "--rl-learning-rate",
        dest="rl_learning_rate",
        type=float,
        default=float(os.environ.get("MINT_TEST_RL_LR", "2e-5")),
        help="RL learning rate passed to optim_step (default: 2e-5; env: MINT_TEST_RL_LR)",
    )
    return parser.parse_args()




def _env_object_id(name: str) -> str | None:
    value = str(os.environ.get(name, "") or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{24}", value):
        raise ValueError(f"{name} must be a 24-character hex object id, got {value!r}")
    return value


def _checkpoint_owner_id() -> str | None:
    return _env_object_id("MINT_TEST_CHECKPOINT_OWNER_ID")


def _create_sampling_client_for_checkpoint(
    service_client: Any,
    *,
    model_path: str,
    base_model: str,
    retry_config: RetryConfig,
) -> Any:
    owner_id = _checkpoint_owner_id()
    if not owner_id:
        return service_client.create_sampling_client(
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )

    holder = service_client.holder
    if not model_path.startswith("mint://"):
        raise ValueError("model_path must start with 'mint://'")
    assert holder._sampling_client_counter is not None
    sampling_session_seq_id = holder._sampling_client_counter
    holder._sampling_client_counter += 1

    async def _create():
        from tinker import types as tinker_types
        from tinker.lib.public_interfaces.sampling_client import SamplingClient

        with holder.aclient(ClientConnectionPoolType.SESSION) as client:
            request = tinker_types.CreateSamplingSessionRequest(
                session_id=holder.get_session_id(),
                sampling_session_seq_id=sampling_session_seq_id,
                model_path=model_path,
                base_model=base_model,
            )
            result = await client.service.create_sampling_session(
                request=request,
                extra_body={"owner_id": owner_id},
            )
        return SamplingClient(holder, sampling_session_id=result.sampling_session_id, retry_config=retry_config)

    return holder.run_coroutine_threadsafe(_create()).result()


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: system prompt file not found: {path!r}. Continuing without it.")
        return ""


args = _parse_args()
TIMEOUT_S = float(args.timeout_s)

# Load API key from .env file
load_dotenv(override=False)

_install_tinker_future_debug()
_patch_trust_remote_code_for_tokenizers(
    enable=(args.base_model == "moonshotai/Moonlight-16B-A3B-Instruct"),
)

print(f"TINKER_BASE_URL={os.environ.get('TINKER_BASE_URL')!r}")
if os.environ.get("TINKER_API_KEY"):
    print("TINKER_API_KEY present (value redacted).")
elif os.environ.get("MINT_API_KEY"):
    print("MINT_API_KEY present (value redacted).")
else:
    print("WARNING: neither TINKER_API_KEY nor MINT_API_KEY is set; client init will fail.")

# Create timestamped experiment directory
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_ROOT = Path(os.environ.get("MINT_TEST_EXPERIMENT_ROOT", "/tmp/mint_rl_test_long_experiments"))
EXPERIMENT_DIR = EXPERIMENT_ROOT / EXPERIMENT_TIMESTAMP
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Experiment directory: {EXPERIMENT_DIR}")
_TIMING_JSONL_PATH = EXPERIMENT_DIR / "timing_events.jsonl"
_TIMING_SUMMARY_JSON_PATH = EXPERIMENT_DIR / "timing_summary.json"
_TIMING_SUMMARY_MD_PATH = EXPERIMENT_DIR / "timing_summary.md"
atexit.register(_write_timing_reports)

# Create the service client
service_client = _time_call(mint.ServiceClient, stage_name="create_service_client")
_dbg(f"client_session_id={getattr(getattr(service_client, 'holder', None), '_session_id', None)!r}")

# List available models
print("Connected to MinT server!")
print("\nAvailable models:")
try:
    capabilities = _time_call(
        service_client.get_server_capabilities,
        stage_name="get_server_capabilities",
    )
    for model in capabilities.supported_models:
        print(f"  - {model.model_name}")
except Exception as e:
    print(f"  Error: {e}")

BASE_MODEL = args.base_model


inference_only = bool(args.inference_only)

if int(getattr(args, "lora_rank", 0)) > 0:
    LORA_RANK = int(args.lora_rank)
else:
    # Megatron LoRA sharding requires LoRA rank divisible by TP.
    # Default K2 rank is 64 (divisible by TP=64 and TP=32).
    LORA_RANK = 64 if BASE_MODEL.startswith("moonshotai/Kimi-K2-") else 16

train_mlp = args.train_mlp
if train_mlp is None:
    # K2 is MoE; MLP LoRA export+per-expert expansion is heavy and not required to validate end-to-end RL plumbing.
    train_mlp = BASE_MODEL not in {"moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2-Instruct"}
    # train_mlp = True

train_attn = True if args.train_attn is None else bool(args.train_attn)
train_unembed = True if args.train_unembed is None else bool(args.train_unembed)

# Tokenizer comes from TrainingClient; SamplingClient does not expose tokenizer.
# For Qwen3 models, tokenizer is shared across sizes; use 0.6B as a lightweight tokenizer source.
tokenizer_source_model = BASE_MODEL
if inference_only and BASE_MODEL.startswith("Qwen/Qwen3-"):
    tokenizer_source_model = "Qwen/Qwen3-0.6B"

use_hf_tokenizer = BASE_MODEL.startswith("moonshotai/Kimi-K2-") or (
    inference_only and BASE_MODEL.startswith("Qwen/Qwen3-")
)
if use_hf_tokenizer:
    from transformers import AutoTokenizer

    tokenizer_model = BASE_MODEL if BASE_MODEL.startswith("moonshotai/Kimi-K2-") else tokenizer_source_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    print(f"Tokenizer loaded from HF: {tokenizer_model}")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size:,} tokens")
else:
    tokenizer_training_client = _time_call(
        lambda: service_client.create_lora_training_client(
            base_model=tokenizer_source_model,
            rank=LORA_RANK,
            train_mlp=train_mlp if tokenizer_source_model == BASE_MODEL else False,
            train_attn=train_attn if tokenizer_source_model == BASE_MODEL else True,
            train_unembed=train_unembed if tokenizer_source_model == BASE_MODEL else False,
        ),
        stage_name="create_tokenizer_training_client",
        extra={"tokenizer_source_model": tokenizer_source_model},
    )
    print(f"Tokenizer training client created for: {tokenizer_source_model}")
    _dbg(f"tokenizer_training_model_id={getattr(tokenizer_training_client, 'model_id', None)!r}")

    tokenizer = _time_call(
        tokenizer_training_client.get_tokenizer,
        stage_name="get_tokenizer",
        extra={"tokenizer_source_model": tokenizer_source_model},
    )
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size:,} tokens")

training_client = None
if not inference_only:
    training_client = _time_call(
        lambda: service_client.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=LORA_RANK,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        ),
        stage_name="create_training_client",
    )
    print(f"Training client created for: {BASE_MODEL}")
    _dbg(f"training_model_id={getattr(training_client, 'model_id', None)!r}")

random.seed(42)  # For reproducibility

SYSTEM_PROMPT = _read_text_file(args.system_prompt_path).strip()
if SYSTEM_PROMPT:
    print(f"Loaded system prompt from: {args.system_prompt_path} ({len(SYSTEM_PROMPT)} chars)")
else:
    print("System prompt is empty; prompts will not be extended.")

print(f"Prompt mode: {'chat_template' if hasattr(tokenizer, 'apply_chat_template') else 'plain'}")

def _compute_stop_token_ids(tokenizer: Any) -> list[int]:
    # Kimi models commonly terminate assistant messages with <|im_end|> (or [EOT]),
    # not [EOS]. Using only eos_token_id can lead to generation running until max_tokens.
    out: list[int] = []
    # Qwen-family tokenizers often have both <|im_end|> and <|endoftext|> as single-token sentinels;
    # include both so we stop even if the model terminates with the other one.
    for tok in ("<|im_end|>", "<|endoftext|>", "[EOT]"):
        tid = None
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and (unk is None or tid != unk):
            out.append(int(tid))
            continue
        try:
            ids = tokenizer.encode(tok, add_special_tokens=False)
        except Exception:
            ids = None
        if isinstance(ids, list) and len(ids) == 1:
            try:
                out.append(int(ids[0]))
            except Exception:
                pass
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        out.append(int(eos))
    seen: set[int] = set()
    dedup: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup

STOP_TOKEN_IDS = _compute_stop_token_ids(tokenizer)
print(f"Stop token ids: {STOP_TOKEN_IDS}")

PROGRESS_TIMEOUT_S = float(os.environ.get("MINT_TEST_PROGRESS_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
SAMPLING_RETRY_CONFIG = RetryConfig(progress_timeout=PROGRESS_TIMEOUT_S)
print(f"Sampling progress_timeout_s: {PROGRESS_TIMEOUT_S}")


PROMPT_TARGET_TOKENS = int(getattr(args, "prompt_target_tokens", 0))
PROMPT_MIN_TOKENS = int(args.prompt_min_tokens) if PROMPT_TARGET_TOKENS <= 0 else 0
_LONG_PREFIX_TEXT: str = ""

def _pick_single_token_pad_id(tokenizer: Any) -> int:
    candidates = [" ", "\n", " a", " the", "x"]
    for s in candidates:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            continue
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int) and ids[0] >= 0:
            return int(ids[0])
    raise RuntimeError("Could not find a stable single-token pad string for this tokenizer")


_PROMPT_PAD_TOKEN_ID: int | None = None


def _pad_prompt_tokens_exact(tokens: list[int], *, target: int) -> list[int]:
    if target <= 0:
        return tokens
    if len(tokens) > target:
        raise RuntimeError(f"Prompt already longer than target: len={len(tokens)} target={target}")
    if len(tokens) == target:
        return tokens

    global _PROMPT_PAD_TOKEN_ID
    if _PROMPT_PAD_TOKEN_ID is None:
        _PROMPT_PAD_TOKEN_ID = _pick_single_token_pad_id(tokenizer)

    insert_at = 0
    bos = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos, int) and tokens and tokens[0] == bos:
        insert_at = 1

    need = target - len(tokens)
    return tokens[:insert_at] + [_PROMPT_PAD_TOKEN_ID] * need + tokens[insert_at:]


def _compute_long_prefix_text(*, dummy_question: str) -> str:
    if PROMPT_MIN_TOKENS <= 0:
        return ""

    if hasattr(tokenizer, "apply_chat_template"):
        messages: list[dict[str, str]] = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": dummy_question})
        base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        base = f"{SYSTEM_PROMPT}\n\nQuestion: {dummy_question}\nAnswer:" if SYSTEM_PROMPT else f"Question: {dummy_question}\nAnswer:"

    base_tokens = tokenizer.encode(base)
    if len(base_tokens) >= PROMPT_MIN_TOKENS:
        return ""

    filler_piece = (
        "Context: This is filler text for long-context stress testing. "
        "Do not answer based on this filler.\n"
    )
    piece_tokens = tokenizer.encode(filler_piece)
    if not piece_tokens:
        raise RuntimeError("tokenizer produced 0 tokens for filler_piece")

    need = PROMPT_MIN_TOKENS - len(base_tokens)
    n = (need + len(piece_tokens) - 1) // len(piece_tokens)
    prefix = filler_piece * max(1, n)

    # Verify and top up if join effects reduce token count.
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": f"{prefix}\n{dummy_question}"})
        padded = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        q = f"{prefix}\n{dummy_question}"
        padded = f"{SYSTEM_PROMPT}\n\nQuestion: {q}\nAnswer:" if SYSTEM_PROMPT else f"Question: {q}\nAnswer:"

    padded_tokens = tokenizer.encode(padded)
    while len(padded_tokens) < PROMPT_MIN_TOKENS:
        prefix += filler_piece
        if hasattr(tokenizer, "apply_chat_template"):
            messages = []
            if SYSTEM_PROMPT:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": f"{prefix}\n{dummy_question}"})
            padded = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            q = f"{prefix}\n{dummy_question}"
            padded = f"{SYSTEM_PROMPT}\n\nQuestion: {q}\nAnswer:" if SYSTEM_PROMPT else f"Question: {q}\nAnswer:"
        padded_tokens = tokenizer.encode(padded)

    return prefix


if PROMPT_MIN_TOKENS > 0:
    _LONG_PREFIX_TEXT = _compute_long_prefix_text(dummy_question="What is 11 * 11?")
    if _LONG_PREFIX_TEXT:
        dummy_question = "What is 11 * 11?"
        if hasattr(tokenizer, "apply_chat_template"):
            msgs: list[dict[str, str]] = []
            if SYSTEM_PROMPT:
                msgs.append({"role": "system", "content": SYSTEM_PROMPT})
            msgs.append({"role": "user", "content": dummy_question})
            base_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            msgs[-1]["content"] = f"{_LONG_PREFIX_TEXT}\n\n{dummy_question}"
            padded_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            base_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {dummy_question}\nAnswer:" if SYSTEM_PROMPT else f"Question: {dummy_question}\nAnswer:"
            q = f"{_LONG_PREFIX_TEXT}\n\n{dummy_question}"
            padded_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {q}\nAnswer:" if SYSTEM_PROMPT else f"Question: {q}\nAnswer:"
        print(
            f"Prompt padding enabled: base_tokens={len(tokenizer.encode(base_prompt))} "
            f"padded_tokens={len(tokenizer.encode(padded_prompt))} target_min={PROMPT_MIN_TOKENS}"
        )
elif PROMPT_TARGET_TOKENS > 0:
    dummy_question = "What is 11 * 11?"
    if hasattr(tokenizer, "apply_chat_template"):
        msgs: list[dict[str, str]] = []
        if SYSTEM_PROMPT:
            msgs.append({"role": "system", "content": SYSTEM_PROMPT})
        msgs.append({"role": "user", "content": dummy_question})
        base_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        base_prompt = (
            f"{SYSTEM_PROMPT}\n\nQuestion: {dummy_question}\nAnswer:"
            if SYSTEM_PROMPT
            else f"Question: {dummy_question}\nAnswer:"
        )
    base_tokens = tokenizer.encode(base_prompt)
    padded_tokens = _pad_prompt_tokens_exact(list(base_tokens), target=PROMPT_TARGET_TOKENS)
    print(
        f"Prompt target padding enabled: base_tokens={len(base_tokens)} "
        f"padded_tokens={len(padded_tokens)} target={PROMPT_TARGET_TOKENS}"
    )


def _save_weights_and_get_sampling_client_with_retry(
    service_client: Any,
    training_client: Any,
    base_model: str,
    name: str,
    *,
    max_retries: int = 4,
    backoff_s: float = 1.0,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            save_fut = training_client.save_weights_for_sampler(name=name)
            _dbg(f"save_weights_for_sampler submitted name={name!r}")
            sampling_path = _result_with_heartbeat(
                save_fut,
                label="SaveWeightsForSampler",
                timeout_s=TIMEOUT_S,
                stage_name="save_weights_for_sampler",
                extra={"name": name},
            ).path
            sampling_client = _time_call(
                lambda: _create_sampling_client_for_checkpoint(
                    service_client,
                    model_path=sampling_path,
                    base_model=base_model,
                    retry_config=SAMPLING_RETRY_CONFIG,
                ),
                stage_name="create_sampling_client",
                extra={"name": name, "model_path": sampling_path},
            )
            _dbg(f"sampling_session_id={getattr(sampling_client, '_sampling_session_id', None)!r}")
            return sampling_client
        except Exception as e:
            last_exc = e
            print(
                f"save_weights_for_sampler({name!r}) failed: {e} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                time.sleep(min(backoff_s * (2**attempt), 10.0))
    raise RuntimeError(f"Failed to create sampling client for {name!r} after {max_retries} attempts") from last_exc


def extract_answer(response: str) -> str | None:
    """Extract the first numeric answer from response."""
    numbers = re.findall(r"\d+", response)
    return numbers[-1] if numbers else None


def generate_rl_problem():
    """Generate multiplication problems for RL."""
    a = random.randint(10, 9999)
    b = random.randint(10, 9999)
    return f"What is {a} * {b}?", str(a * b)


def compute_reward(response: str, correct_answer: str) -> float:
    """Reward function: 1.0 if correct, 0.0 otherwise."""
    extracted = extract_answer(response)
    reward = 1.0 if extracted == correct_answer else 0.0
    reward -= 0.001 * len(response)
    if extracted is None:
        return reward
    try:
        reward += math.exp(
            -100.0 * abs(float(extracted) - float(correct_answer)) / float(correct_answer)
        )
    except Exception:
        pass
    return reward


def _build_prompt(question: str) -> str:
    if _LONG_PREFIX_TEXT:
        question = f"{_LONG_PREFIX_TEXT}\n\n{question}"
    if hasattr(tokenizer, "apply_chat_template"):
        messages: list[dict[str, str]] = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    if SYSTEM_PROMPT:
        return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"

def _build_prompt_tokens(question: str) -> list[int]:
    prompt_text = _build_prompt(question)
    tokens = tokenizer.encode(prompt_text)
    if PROMPT_TARGET_TOKENS > 0:
        tokens = _pad_prompt_tokens_exact(list(tokens), target=PROMPT_TARGET_TOKENS)
    return tokens


def _token_count_text(text: str) -> int:
    return len(tokenizer.encode(text))


def _as_int_list(xs: list[Any]) -> list[int]:
    return [int(x) for x in xs]


def _as_float_list(xs: list[Any]) -> list[float]:
    out: list[float] = []
    for x in xs:
        v = float(x)
        if not math.isfinite(v):
            v = 0.0
        out.append(v)
    return out


def _decode_with_visible_special_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    """
    Decode token_ids into text while making special/non-text tokens visible.

    This is diagnostic-only; do not use for reward extraction because special-token
    strings may contain digits (e.g. "reserved_special_token_0").
    """
    def _has_visible_glyph(s: str) -> bool:
        for ch in s:
            if ch.isspace():
                continue
            if unicodedata.category(ch).startswith("C"):
                continue
            if not ch.isprintable():
                continue
            return True
        return False

    parts: list[str] = []
    for tid in token_ids:
        s = tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if s and _has_visible_glyph(s):
            parts.append(s)
            continue

        if s:
            parts.append(f"<INVIS_TOKEN:{int(tid)}:{s!r}>")
            continue

        tok = None
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            try:
                tok = tokenizer.convert_ids_to_tokens(int(tid), skip_special_tokens=False)
            except Exception:
                tok = None
        parts.append(f"<NONDECODE_TOKEN:{int(tid)}:{tok}>" if tok else f"<NONDECODE_TOKEN:{int(tid)}>")
    return "".join(parts)


def _summarize_loss_inputs(loss_fn_inputs: dict[str, Any]) -> dict[str, Any]:
    def _len(x: Any) -> int | None:
        try:
            return len(x)
        except Exception:
            return None

    return {
        "keys": sorted(list(loss_fn_inputs.keys())),
        "target_tokens_len": _len(loss_fn_inputs.get("target_tokens")),
        "weights_len": _len(loss_fn_inputs.get("weights")),
        "logprobs_len": _len(loss_fn_inputs.get("logprobs")),
        "advantages_len": _len(loss_fn_inputs.get("advantages")),
        "target_tokens_type": type(loss_fn_inputs.get("target_tokens")).__name__,
        "weights_type": type(loss_fn_inputs.get("weights")).__name__,
        "logprobs_type": type(loss_fn_inputs.get("logprobs")).__name__,
        "advantages_type": type(loss_fn_inputs.get("advantages")).__name__,
    }


def _validate_loss_inputs(loss_fn_inputs: dict[str, Any]) -> tuple[bool, str]:
    required = ["target_tokens", "weights", "logprobs", "advantages"]
    missing = [k for k in required if k not in loss_fn_inputs]
    if missing:
        return False, f"missing keys: {missing}"

    lengths = {
        k: (len(loss_fn_inputs[k]) if hasattr(loss_fn_inputs[k], "__len__") else None)
        for k in required
    }
    if any(v is None for v in lengths.values()):
        return False, f"non-sequence values in loss_fn_inputs: {lengths}"

    uniq = {v for v in lengths.values() if v is not None}
    if len(uniq) != 1:
        return False, f"length mismatch: {lengths}"

    if lengths["target_tokens"] == 0:
        return False, "empty target_tokens"

    return True, "ok"


# Demo
print("Reward function demo:")
print("RL problems: 10-9999 * 10-9999")
print()
q, a = generate_rl_problem()
print(f"Question: {q}, Answer: {a}")
print(f"  '{a}' → reward = {compute_reward(a, a)}")
print(f"  '999' → reward = {compute_reward('999', a)}")

# RL Configuration
NUM_RL_STEPS = int(args.num_rl_steps)
BATCH_SIZE = int(args.batch_size)
GROUP_SIZE = int(args.group_size)
MAX_TOKENS = int(args.max_tokens)
RL_LEARNING_RATE = float(args.rl_learning_rate)
TEMPERATURE = float(args.temperature)

rl_metrics = []
print("Starting RL training (no SFT)...")
print(
    f"Config: {NUM_RL_STEPS} steps, {BATCH_SIZE} problems/batch, {GROUP_SIZE} samples/problem"
)
print()

if inference_only:
    print("Inference-only mode: sampling on base model (no LoRA training).")
    sampling_client = _time_call(
        lambda: service_client.create_sampling_client(
            base_model=BASE_MODEL,
            retry_config=SAMPLING_RETRY_CONFIG,
        ),
        stage_name="create_base_sampling_client",
    )

    all_rewards: list[float] = []
    all_lat_s: list[float] = []

    for step in range(NUM_RL_STEPS):
        step_t0 = time.time()
        problems = [generate_rl_problem() for _ in range(BATCH_SIZE)]
        for question, answer in problems:
            prompt_tokens = _build_prompt_tokens(question)
            prompt_input = types.ModelInput.from_ints(prompt_tokens)

            if args.print_token_counts:
                print(f"[infer step {step + 1}] prompt_tokens={len(prompt_tokens)}")

            t0 = time.time()
            sample_result = sampling_client.sample(
                prompt=prompt_input,
                num_samples=GROUP_SIZE,
                sampling_params=types.SamplingParams(
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stop=STOP_TOKEN_IDS,
                ),
            )
            sample_result = _result_with_heartbeat(
                sample_result,
                label="Sample",
                timeout_s=TIMEOUT_S,
                stage_name="sample",
                step_idx=step,
                extra={"mode": "inference_only"},
            )
            all_lat_s.append(time.time() - t0)

            rewards = []
            for seq_idx, seq in enumerate(sample_result.sequences):
                response_text_for_reward = tokenizer.decode(
                    seq.tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response_text = _decode_with_visible_special_tokens(tokenizer, list(seq.tokens))
                reward = compute_reward(response_text_for_reward, answer)
                rewards.append(reward)
                if args.print_token_counts:
                    print(
                        f"[infer step {step + 1}] sample[{seq_idx}] output_tokens={len(seq.tokens)} "
                        f"reward={reward}"
                    )
                if args.print_sample_text:
                    print(
                        f"[infer step {step + 1}] sample[{seq_idx}] decoded={response_text!r}"
                    )
            all_rewards.extend(rewards)

        if all_rewards:
            step_mean = sum(all_rewards) / len(all_rewards)
            print(f"[infer step {step + 1}] running_mean_reward={step_mean:.3f} (n={len(all_rewards)})")
        _append_timing_event(
            "inference_step_total",
            elapsed_s=time.time() - step_t0,
            status="ok",
            step_idx=step,
            extra={"num_problems": len(problems)},
        )

    if all_lat_s:
        lat_p50 = sorted(all_lat_s)[len(all_lat_s) // 2]
        print(f"inference_only requests={len(all_lat_s)} latency_p50_s={lat_p50:.2f} base_model={BASE_MODEL!r}")
    if all_rewards:
        mean_reward = sum(all_rewards) / len(all_rewards)
        print(f"inference_only mean_reward={mean_reward:.3f} (n={len(all_rewards)}) base_model={BASE_MODEL!r}")
    raise SystemExit(0)

for step in range(NUM_RL_STEPS):
    step_t0 = time.time()
    try:
        rl_sampling_client = _save_weights_and_get_sampling_client_with_retry(
            service_client,
            training_client,
            BASE_MODEL,
            name=f"rl-step-{step}",
        )
    except Exception as e:
        print(f"[step {step + 1}] Failed to create sampling client: {e}")
        break

    problems = [generate_rl_problem() for _ in range(BATCH_SIZE)]
    training_datums = []
    all_rewards = []

    for question, answer in problems:
        prompt_tokens = _build_prompt_tokens(question)
        prompt_input = types.ModelInput.from_ints(prompt_tokens)

        if args.print_token_counts:
            print(
                f"[step {step + 1}] prompt_tokens={len(prompt_tokens)} "
                f"(question={question!r}, answer_tokens={_token_count_text(answer)})"
            )

        sample_result = rl_sampling_client.sample(
            prompt=prompt_input,
            num_samples=GROUP_SIZE,
            sampling_params=types.SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop=STOP_TOKEN_IDS,
            ),
        )
        sample_result = _result_with_heartbeat(
            sample_result,
            label="Sample",
            timeout_s=TIMEOUT_S,
            stage_name="sample",
            step_idx=step,
            extra={"mode": "training"},
        )

        group_rewards = []
        group_responses = []
        group_logprobs = []

        for seq_idx, seq in enumerate(sample_result.sequences):
            response_text_for_reward = tokenizer.decode(
                seq.tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response_text = _decode_with_visible_special_tokens(tokenizer, list(seq.tokens))
            #print(response_text)

            eos_id = tokenizer.eos_token_id
            eos_pos = list(seq.tokens).index(eos_id) if eos_id in seq.tokens else None
            hit_max = len(seq.tokens) >= MAX_TOKENS

            reward = compute_reward(response_text_for_reward, answer)
            group_rewards.append(reward)
            group_responses.append(list(seq.tokens))
            group_logprobs.append(list(seq.logprobs) if seq.logprobs else [0.0] * len(seq.tokens))

            if args.print_token_counts:
                print(
                    f"[step {step + 1}] sample[{seq_idx}] output_tokens={len(seq.tokens)} "
                    f"reward={reward} extracted={extract_answer(response_text_for_reward)!r}"
                )
            if args.print_sample_text:
                print(f"[step {step + 1}] sample[{seq_idx}] decoded={response_text!r}")

        all_rewards.extend(group_rewards)

        mean_reward = sum(group_rewards) / len(group_rewards)
        advantages = [r - mean_reward for r in group_rewards]

        if all(a == 0 for a in advantages):
            continue

        for response_tokens, logprobs, adv in zip(group_responses, group_logprobs, advantages):
            if len(response_tokens) == 0:
                continue

            full_tokens = prompt_tokens + response_tokens
            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]

            weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(
                response_tokens
            )
            full_logprobs = [0.0] * (len(prompt_tokens) - 1) + list(logprobs)
            full_advantages = [0.0] * (len(prompt_tokens) - 1) + [adv] * len(
                response_tokens
            )

            loss_fn_inputs = dict(
                target_tokens=_as_int_list(target_tokens),
                weights=_as_float_list(weights),
                logprobs=_as_float_list(full_logprobs),
                advantages=_as_float_list(full_advantages),
            )

            ok, msg = _validate_loss_inputs(loss_fn_inputs)
            if not ok:
                print(
                    f"[step {step + 1}] Skipping invalid datum ({msg}). Summary: "
                    f"{_summarize_loss_inputs(loss_fn_inputs)}"
                )
                continue

            training_datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs=loss_fn_inputs,
                )
            )

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    accuracy = (
        sum(1 for r in all_rewards if r > 0) / len(all_rewards)
        if all_rewards
        else 0.0
    )

    if training_datums:
        fwdbwd_future = training_client.forward_backward(
            training_datums,
            loss_fn="importance_sampling",
        )
        _result_with_heartbeat(
            fwdbwd_future,
            label="ForwardBackward",
            timeout_s=TIMEOUT_S,
            stage_name="forward_backward",
            step_idx=step,
            extra={"num_datums": len(training_datums)},
        )

        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=RL_LEARNING_RATE),
        )
        _result_with_heartbeat(
            optim_future,
            label="OptimStep",
            timeout_s=TIMEOUT_S,
            stage_name="optim_step",
            step_idx=step,
            extra={"num_datums": len(training_datums)},
        )

    rl_metrics.append(
        {
            "step": step,
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "num_datums": len(training_datums),
        }
    )
    _append_timing_event(
        "rl_step_total",
        elapsed_s=time.time() - step_t0,
        status="ok",
        step_idx=step,
        extra={"num_datums": len(training_datums), "avg_reward": round(avg_reward, 6)},
    )

    print(
        f"Step {step + 1:2d}/{NUM_RL_STEPS}: Accuracy = {accuracy:5.1%}, Avg Reward = {avg_reward:.3f}"
    )

print("\nRL training complete!")

# Get final RL model
final_path = training_client.save_weights_for_sampler(
    name="arithmetic-rl-final",
)
final_path = _result_with_heartbeat(
    final_path,
    label="SaveWeightsForSampler",
    timeout_s=TIMEOUT_S,
    stage_name="save_weights_for_sampler",
    extra={"name": "arithmetic-rl-final"},
).path
final_client = _time_call(
    lambda: _create_sampling_client_for_checkpoint(
        service_client,
        model_path=final_path,
        base_model=BASE_MODEL,
        retry_config=SAMPLING_RETRY_CONFIG,
    ),
    stage_name="create_sampling_client",
    extra={"name": "arithmetic-rl-final", "model_path": final_path},
)
_dbg(f"final_sampling_session_id={getattr(final_client, '_sampling_session_id', None)!r}")

# Test on problems (10-199 range)
rl_test_problems = [
    ("What is 123 * 45?", "5535"),
    ("What is 67 * 189?", "12663"),
    ("What is 156 * 78?", "12168"),
]

print("Testing RL-only model (10-199 range):")
print("=" * 50)
rl_correct = 0

for question, correct in rl_test_problems:
    prompt_text = _build_prompt(question)
    if args.print_token_counts:
        prompt_tokens = _build_prompt_tokens(question)
        print(
            f"[eval] prompt_tokens={len(prompt_tokens)} question={question!r} "
            f"answer_tokens={_token_count_text(correct)}"
        )

    prompt_input = types.ModelInput.from_ints(_build_prompt_tokens(question))

    fut = final_client.sample(
        prompt=prompt_input,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=16,
            temperature=0.0,
            stop=STOP_TOKEN_IDS,
        ),
    )
    result = _result_with_heartbeat(
        fut,
        label="EvalSample",
        timeout_s=TIMEOUT_S,
        stage_name="eval_sample",
        extra={"question": question},
    )

    response_for_reward = tokenizer.decode(
        result.sequences[0].tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = _decode_with_visible_special_tokens(tokenizer, list(result.sequences[0].tokens))
    extracted = extract_answer(response_for_reward)
    is_correct = extracted == correct
    if is_correct:
        rl_correct += 1

    print(f"Q: {question}")
    print(
        f"A: {response.strip()} (extracted: {extracted}, correct: {correct}) "
        f"[{'PASS' if is_correct else 'FAIL'}]"
    )
    print()

print(f"RL Accuracy: {rl_correct}/{len(rl_test_problems)}")

if rl_metrics:
    if pd is not None:
        df_rl = pd.DataFrame(rl_metrics)
        csv_path = (EXPERIMENT_DIR / "rl_metrics.csv").resolve()
        df_rl.to_csv(csv_path, index=False)
        print(f"Saved RL metrics CSV to: {csv_path}")
    else:
        print("Skipping rl_metrics.csv: pandas not installed")

if rl_metrics:
    if plt is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        rl_steps = [m["step"] + 1 for m in rl_metrics]
        rl_accuracy = [m["accuracy"] for m in rl_metrics]

        ax.plot(
            rl_steps,
            rl_accuracy,
            "g-o",
            linewidth=2,
            markersize=8,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("RL Training Accuracy (No SFT)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plot_path = (EXPERIMENT_DIR / "rl_training_results.png").resolve()
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
    else:
        print("Skipping rl_training_results.png: matplotlib not installed")

if args.save_ckpt:
    # Save final RL checkpoint
    save_state_future = training_client.save_state(name="arithmetic-rl-final")
    rl_checkpoint = _result_with_heartbeat(
        save_state_future,
        label="SaveState",
        timeout_s=TIMEOUT_S,
        stage_name="save_state",
        extra={"name": "arithmetic-rl-final"},
    )
    print(f"Final checkpoint: {rl_checkpoint.path}")

    ckpt_info = {
        "experiment_dir": str(EXPERIMENT_DIR.resolve()),
        "base_model": BASE_MODEL,
        "system_prompt_path": args.system_prompt_path,
        "final_sampling_path": final_path,
        "final_resume_path": rl_checkpoint.path,
        "metrics_csv": str((EXPERIMENT_DIR / "rl_metrics.csv").resolve()),
        "plot_path": str((EXPERIMENT_DIR / "rl_training_results.png").resolve()),
        "timing_events_jsonl": str((_TIMING_JSONL_PATH or (EXPERIMENT_DIR / "timing_events.jsonl")).resolve()),
        "timing_summary_json": str((_TIMING_SUMMARY_JSON_PATH or (EXPERIMENT_DIR / "timing_summary.json")).resolve()),
        "timing_summary_md": str((_TIMING_SUMMARY_MD_PATH or (EXPERIMENT_DIR / "timing_summary.md")).resolve()),
    }
    ckpt_path = (EXPERIMENT_DIR / "checkpoints.json").resolve()
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ckpt_info, f, indent=2, ensure_ascii=False)
    print(f"Wrote checkpoint links to: {ckpt_path}")

    print("\nTo resume training later:")
    print(
        f"  client = service_client.create_training_client_from_state('{rl_checkpoint.path}')"
    )
else:
    print("Skipping checkpoint save (--no-save-ckpt).")
