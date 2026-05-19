import anyio
import httpx

import mint_server.logging_context as logging_context


def test_issue_217_gateway_forward_json_reuses_client(monkeypatch):
    import mint_server.gateway as gw

    class _DummyAsyncClient:
        created = 0
        closed = 0

        def __init__(self, *args, **kwargs):
            type(self).created += 1
            self.is_closed = False

        async def request(self, method, path, headers=None, json=None, timeout=None):
            return httpx.Response(200, json={"method": method, "path": path})

        async def aclose(self):
            if not self.is_closed:
                self.is_closed = True
                type(self).closed += 1

    gw._http_clients.clear()
    monkeypatch.setattr(gw.httpx, "AsyncClient", _DummyAsyncClient)

    async def _run():
        upstream = gw.Upstream(alias="u1", base_url="http://example.com", auth_mode="none", api_key=None)
        await gw.forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/asample",
            incoming_headers={},
            json_body={"x": 1},
            timeout_s=1.0,
        )
        await gw.forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/retrieve_future",
            incoming_headers={},
            json_body={"request_id": "rid"},
            timeout_s=1.0,
        )
        assert _DummyAsyncClient.created == 1
        await gw.close_http_clients()
        assert _DummyAsyncClient.closed == 1

    anyio.run(_run)


def test_issue_217_gateway_forwards_trace_headers(monkeypatch):
    import mint_server.gateway as gw

    captured: dict[str, object] = {}

    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.is_closed = False

        async def request(self, method, path, headers=None, json=None, timeout=None):
            captured["headers"] = dict(headers or {})
            return httpx.Response(200, json={"ok": True})

        async def aclose(self):
            self.is_closed = True

    gw._http_clients.clear()
    monkeypatch.setattr(gw.httpx, "AsyncClient", _DummyAsyncClient)

    prev_trace_id = logging_context.get_trace_id()
    try:
        logging_context.set_trace_id("b" * 32)

        async def _run():
            upstream = gw.Upstream(alias="u1", base_url="http://example.com", auth_mode="none", api_key=None)
            await gw.forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/asample",
                incoming_headers={"traceparent": "00-" + ("a" * 32) + "-" + ("1" * 16) + "-01"},
                json_body={"x": 1},
                timeout_s=1.0,
            )

        anyio.run(_run)
    finally:
        logging_context.set_trace_id(prev_trace_id)
        anyio.run(gw.close_http_clients)

    assert captured["headers"]["traceparent"] == "00-" + ("a" * 32) + "-" + ("1" * 16) + "-01"
    assert captured["headers"]["X-Trace-Id"] == "b" * 32
