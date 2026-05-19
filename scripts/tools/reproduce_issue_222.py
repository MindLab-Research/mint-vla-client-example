import sys


ISSUE_NUMBER = 222


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    print(f"issue={ISSUE_NUMBER}")

    from mint_server.backend.vllm_stop import vllm_stop_kwargs

    # The prod issue shows trained models emitting a literal "\\n\\n" suffix (two chars: backslash+n)
    # after a JSON object, so stop="\n\n" (real newlines) does not terminate.
    #
    # Fix strategy: treat stop containing real newline characters as also matching the
    # literal backslash-n form, so stop="\n\n" expands to stop=["\n\n", "\\n\\n"].
    out = vllm_stop_kwargs("\n\n", default_stop_token_ids=None)
    stop = out.get("stop")
    if stop != ["\n\n", "\\n\\n"]:
        return _fail(f"expected stop expansion ['\\\\n\\\\n','\\\\\\\\n\\\\\\\\n'], got {stop!r}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

