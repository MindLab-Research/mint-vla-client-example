from __future__ import annotations


def flatten_encoded_text_chunks(model_input: dict) -> list[int]:
    chunks = model_input.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("model_input.chunks must be a list")

    out: list[int] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("model_input.chunks entries must be dicts")
        if chunk.get("type") not in (None, "encoded_text"):
            raise ValueError(f"Unsupported chunk type: {chunk.get('type')}")
        toks = chunk.get("tokens")
        if not isinstance(toks, list):
            raise ValueError("chunk.tokens must be a list[int]")
        out.extend(toks)
    return out
