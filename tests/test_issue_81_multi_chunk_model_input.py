from tinker_server.model_input_utils import flatten_encoded_text_chunks


def test_issue_81_flattens_multi_chunk_model_input() -> None:
    model_input = {
        "chunks": [
            {"type": "encoded_text", "tokens": [1, 2]},
            {"type": "encoded_text", "tokens": [3]},
        ]
    }
    assert flatten_encoded_text_chunks(model_input) == [1, 2, 3]


def test_issue_81_ignores_non_encoded_chunks() -> None:
    model_input = {
        "chunks": [
            {"type": "image", "data": "Zm9v", "format": "png", "expected_tokens": 8},
            {"type": "encoded_text", "tokens": [3]},
        ]
    }
    try:
        flatten_encoded_text_chunks(model_input)
    except ValueError as e:
        assert "Unsupported chunk type" in str(e)
    else:
        raise AssertionError("Expected ValueError for non-encoded chunk type")

