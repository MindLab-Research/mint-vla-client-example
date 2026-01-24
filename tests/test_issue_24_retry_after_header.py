from tinker_server.futures_utils import pending_future_http_response


def test_issue_24_sets_retry_after_header() -> None:
    pending = pending_future_http_response()
    assert pending.status_code == 408
    assert pending.headers.get("Retry-After") == "1"
    assert pending.body == {"queue_state": "active"}
