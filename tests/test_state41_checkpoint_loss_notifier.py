from __future__ import annotations

import json
import subprocess

import pytest

from scripts.notify import state41_checkpoint_loss_notifier as notifier


def _event(step: int = 5000, loss: float = 0.125) -> dict:
    return {
        "contract": notifier.EVENT_CONTRACT,
        "ready_for_notification": True,
        "step": step,
        "loss": loss,
        "checkpoint_kind": "periodic_sampler",
        "checkpoint_path": f"run_step{step}",
        "sampler": {"path": f"mint://run_step{step}"},
        "metrics": {"loss:mean": loss},
        "created_unix_seconds": 1.0,
    }


def test_event_validation_requires_checkpoint_and_exact_step_loss() -> None:
    assert notifier.validate_event(_event())["step"] == 5000
    with pytest.raises(ValueError, match="5K boundary"):
        notifier.validate_event(_event(step=5001))
    mismatched = _event()
    mismatched["metrics"]["loss:mean"] = 0.5
    with pytest.raises(ValueError, match="loss mismatch"):
        notifier.validate_event(mismatched)
    wrong_path = _event()
    wrong_path["checkpoint_path"] = "run_step4999"
    with pytest.raises(ValueError, match="does not end"):
        notifier.validate_event(wrong_path)


def test_event_parser_rejects_duplicates() -> None:
    line = json.dumps(_event())
    with pytest.raises(ValueError, match="duplicate checkpoint event"):
        notifier.parse_events(line + "\n" + line + "\n")


def test_lark_command_is_direct_message_with_idempotency_key() -> None:
    command = notifier.lark_command(
        user_id="ou_recipient",
        identity="user",
        run_id="run",
        event=_event(),
        dry_run=True,
    )
    assert command[:3] == ["lark-cli", "im", "+messages-send"]
    assert command[command.index("--user-id") + 1] == "ou_recipient"
    assert command[command.index("--as") + 1] == "user"
    assert command[-1] == "--dry-run"
    assert "step 5000/100000" in command[command.index("--text") + 1]
    assert notifier.idempotency_key("run", _event()).startswith("state41-5000-")


def test_send_pending_persists_each_success_and_skips_sent_step() -> None:
    ledger = notifier._new_ledger("run", "ou_recipient", "user")
    persisted: list[dict] = []
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"message_id": "om_message"}), stderr=""
        )

    sent = notifier.send_pending(
        [_event()],
        ledger,
        user_id="ou_recipient",
        identity="user",
        run_id="run",
        emit_only=False,
        run_command=run,
        on_sent=lambda value: persisted.append(json.loads(json.dumps(value))),
    )
    assert sent == 1
    assert len(calls) == 1
    assert persisted[-1]["sent"]["5000"]["message_id"] == "om_message"
    assert notifier.send_pending(
        [_event()], ledger,
        user_id="ou_recipient", identity="user", run_id="run",
        emit_only=False, run_command=run,
    ) == 0
    assert len(calls) == 1


def test_send_failure_never_marks_ledger() -> None:
    ledger = notifier._new_ledger("run", "ou_recipient", "bot")

    def fail(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="denied")

    with pytest.raises(RuntimeError, match="Feishu send failed"):
        notifier.send_pending(
            [_event()], ledger,
            user_id="ou_recipient", identity="bot", run_id="run",
            emit_only=False, run_command=fail,
        )
    assert ledger["sent"] == {}
