from __future__ import annotations


def flatten_encoded_text_chunks(model_input: dict) -> list[int]:
    chunks = model_input.get("chunks", [])
    if not isinstance(chunks, list):
        return []

    out: list[int] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") not in (None, "encoded_text"):
            continue
        toks = chunk.get("tokens")
        if not isinstance(toks, list):
            continue
        out.extend(toks)
    return out

