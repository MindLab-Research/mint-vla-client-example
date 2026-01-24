from tinker_server.usage_logger import UsageLogger


def test_usage_summary_operation_counts_are_token_totals(tmp_path):
    logger = UsageLogger(log_dir=str(tmp_path))
    user_id = "u64"

    logger.log(
        user_id=user_id,
        operation_type="sample_prefill",
        model_name="m",
        token_count=10,
        session_id="s",
        request_id="r1",
    )
    logger.log(
        user_id=user_id,
        operation_type="sample_prefill",
        model_name="m",
        token_count=20,
        session_id="s",
        request_id="r2",
    )
    logger.log(
        user_id=user_id,
        operation_type="sample_generation",
        model_name="m",
        token_count=100,
        session_id="s",
        request_id="r3",
    )

    summary = logger.get_user_summary(user_id)
    assert summary["total_tokens"] == 130
    assert summary["operation_counts"]["sample_prefill"] == 30
    assert summary["operation_counts"]["sample_generation"] == 100
