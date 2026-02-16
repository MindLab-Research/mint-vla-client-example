from tinker_server.client_compat import checkpoint_uri, is_tinker_sdk_user_agent


def test_is_tinker_sdk_user_agent():
    assert is_tinker_sdk_user_agent(None) is False
    assert is_tinker_sdk_user_agent("") is False
    assert is_tinker_sdk_user_agent("python-requests/2.31.0") is False
    assert is_tinker_sdk_user_agent("mint/0.1") is False
    assert is_tinker_sdk_user_agent("tinker/0.1") is False
    assert is_tinker_sdk_user_agent("Tinker SDK/0.1") is False
    assert is_tinker_sdk_user_agent("AsyncTinker/Python 0.0.0") is True
    assert is_tinker_sdk_user_agent("Tinker/Python 0.0.0") is True


def test_checkpoint_uri_scheme():
    assert checkpoint_uri("run-1", "0000", prefer_tinker=True) == "tinker://run-1/0000"
    assert checkpoint_uri("run-1", "0000", prefer_tinker=False) == "mint://run-1/0000"
