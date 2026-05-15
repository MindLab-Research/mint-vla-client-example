import tinker_server.backend.api_work_queue as awq
from tinker_server.backend.api_work_queue import ApiWorkQueueThrottleError


def test_issue_324_unwrap_queue_throttle_error_from_ray_wrapper():
    expected = ApiWorkQueueThrottleError(scope="api_key", limit=1, pending=1)

    class _Wrapped(Exception):
        def as_instanceof_cause(self):
            return expected

    unwrapped = awq._unwrap_queue_throttle_error(_Wrapped("wrapped"))
    assert unwrapped is expected
