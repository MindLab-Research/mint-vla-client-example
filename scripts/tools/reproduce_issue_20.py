import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mint_server.sampling_utils import sampled_sequence_from_result  # noqa: E402


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    class Dummy:
        def __init__(self, *, token_ids, stop_reason, logprobs=None, log_probs=None):
            self.token_ids = token_ids
            self.stop_reason = stop_reason
            if logprobs is not None:
                self.logprobs = logprobs
            if log_probs is not None:
                self.log_probs = log_probs

    seq = sampled_sequence_from_result(
        Dummy(token_ids=[42], stop_reason="stop", log_probs=[-0.1])
    )
    if seq.stop_reason != "stop":
        return _fail(f"stop_reason={seq.stop_reason!r} expected 'stop'")
    if seq.logprobs != [-0.1]:
        return _fail(f"logprobs={seq.logprobs!r} expected [-0.1]")

    eos_seq = sampled_sequence_from_result(Dummy(token_ids=[151645], stop_reason=None))
    if eos_seq.stop_reason != "stop":
        return _fail(f"eos stop_reason={eos_seq.stop_reason!r} expected 'stop'")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
