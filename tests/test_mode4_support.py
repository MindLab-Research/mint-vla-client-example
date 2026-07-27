from scripts.eval.mode4_support import (
    acquire_action_session,
    action_session_payload,
    parse_ordered_unique_csv,
)


def test_parse_ordered_unique_rows():
    assert parse_ordered_unique_csv("3,1,3,2", option="--rows") == [3, 1, 2]


def test_external_session_is_not_owned():
    session, owned = acquire_action_session("existing", lambda: "new")
    assert (session, owned) == ("existing", False)


def test_action_session_preserves_model_identity():
    assert action_session_payload(session_id="s", base_model="m", model_path="p", owner_id="o") == {
        "session_id": "s",
        "base_model": "m",
        "model_path": "p",
        "owner_id": "o",
    }
