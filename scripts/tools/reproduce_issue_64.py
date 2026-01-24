import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tinker_server.usage_logger import UsageLogger  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        logger = UsageLogger(log_dir=tmp)
        user_id = "u64"

        logger.log(
            user_id=user_id,
            operation_type="sample_prefill",
            model_name="m",
            token_count=10,
            session_id="s",
            request_id="r1",
        )
        logger.log(
            user_id=user_id,
            operation_type="sample_prefill",
            model_name="m",
            token_count=20,
            session_id="s",
            request_id="r2",
        )
        logger.log(
            user_id=user_id,
            operation_type="sample_generation",
            model_name="m",
            token_count=100,
            session_id="s",
            request_id="r3",
        )

        summary = logger.get_user_summary(user_id)
        total_tokens = summary.get("total_tokens")
        op = summary.get("operation_counts")

        if total_tokens != 130:
            print(f"FAIL: total_tokens={total_tokens!r} expected 130", file=sys.stderr)
            return 1
        if not isinstance(op, dict):
            print(f"FAIL: operation_counts={op!r} expected dict", file=sys.stderr)
            return 1
        if op.get("sample_prefill") != 30:
            print(f"FAIL: sample_prefill={op.get('sample_prefill')!r} expected 30", file=sys.stderr)
            return 1
        if op.get("sample_generation") != 100:
            print(f"FAIL: sample_generation={op.get('sample_generation')!r} expected 100", file=sys.stderr)
            return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
