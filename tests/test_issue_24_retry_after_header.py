from mint_server.futures_utils import pending_future_http_response


def test_issue_24_sets_retry_after_header() -> None:
    pending = pending_future_http_response(retry_after_s=7)
    assert pending.status_code == 408
    assert pending.headers.get("Retry-After") == "7"
    assert pending.body.get("queue_state") == "active"
    assert pending.body.get("retry_after_s") == 7
