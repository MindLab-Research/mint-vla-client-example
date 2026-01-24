import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    megatron_training = repo_root / "tinker_server/backend/megatron_training.py"
    megatron_distributed = repo_root / "tinker_server/backend/megatron_distributed.py"
    utils = repo_root / "tinker_server/model_input_utils.py"

    required = {
        utils: ["def flatten_encoded_text_chunks("],
        megatron_training: [
            "from tinker_server.model_input_utils import flatten_encoded_text_chunks",
            "flatten_encoded_text_chunks(model_input)",
        ],
        megatron_distributed: [
            "from tinker_server.model_input_utils import flatten_encoded_text_chunks",
            "flatten_encoded_text_chunks(model_input)",
        ],
    }

    missing: list[str] = []
    for path, needles in required.items():
        txt = path.read_text(encoding="utf-8")
        for s in needles:
            if s not in txt:
                missing.append(f"{path}: {s}")

    if missing:
        print("FAIL: missing expected strings:", file=sys.stderr)
        for m in missing:
            print(f"- {m}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
