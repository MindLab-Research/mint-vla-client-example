#!/usr/bin/env python3
"""Send fail-closed Feishu notifications for state41 checkpoint/loss events."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


EVENT_CONTRACT = "state41_checkpoint_loss_event_v1"
LEDGER_CONTRACT = "state41_checkpoint_loss_notification_ledger_v1"
DEFAULT_RUN_ID = (
    "state41_gradeA_train95_aug01_qposonly_alora_r16_bs64_4gpu_"
    "contact_pm100_100k_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--ledger-json", type=Path, required=True)
    parser.add_argument("--user-id", required=True, help="Feishu recipient open_id")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--as", dest="identity", choices=("user", "bot"), required=True)
    parser.add_argument("--ssh-host", default="", help="read events from this SSH host")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--emit-only", action="store_true", help="validate and print without sending")
    return parser.parse_args()


def _read_event_text(path: Path, ssh_host: str) -> str:
    if ssh_host:
        result = subprocess.run(
            ["ssh", ssh_host, "cat", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FileNotFoundError(
                f"cannot read {ssh_host}:{path}: {result.stderr.strip()}"
            )
        return result.stdout
    return path.read_text()


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("contract") != EVENT_CONTRACT:
        raise ValueError(f"unsupported checkpoint event contract {event.get('contract')!r}")
    if event.get("ready_for_notification") is not True:
        raise ValueError("checkpoint event is not ready_for_notification")
    step = event.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 5000 or step > 100000:
        raise ValueError(f"invalid checkpoint step {step!r}")
    if step % 5000:
        raise ValueError(f"checkpoint step {step} is not a 5K boundary")
    loss = event.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
        raise ValueError(f"invalid exact-step loss {loss!r}")
    metrics = event.get("metrics")
    if not isinstance(metrics, dict) or "loss:mean" not in metrics:
        raise ValueError("checkpoint event lacks authoritative metrics loss:mean")
    if not math.isclose(float(metrics["loss:mean"]), float(loss), rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            f"checkpoint event loss mismatch {loss!r} != {metrics['loss:mean']!r}"
        )
    checkpoint_path = event.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError("checkpoint event lacks checkpoint_path")
    expected_suffix = f"step{step}"
    if not checkpoint_path.endswith(expected_suffix):
        raise ValueError(
            f"checkpoint path {checkpoint_path!r} does not end in {expected_suffix!r}"
        )
    sampler = event.get("sampler")
    if not isinstance(sampler, dict) or not sampler:
        raise ValueError("checkpoint event lacks successful sampler result")
    return event


def parse_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = validate_event(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid checkpoint event line {line_number}: {error}") from error
        step = int(event["step"])
        if step in seen:
            raise ValueError(f"duplicate checkpoint event step {step}")
        seen.add(step)
        events.append(event)
    return sorted(events, key=lambda value: value["step"])


def message_text(run_id: str, event: dict[str, Any]) -> str:
    return (
        f"state41 Grade-A 训练进度：{run_id}\n"
        f"step {event['step']}/100000\n"
        f"exact-step current loss: {float(event['loss']):.12g}\n"
        f"sampler checkpoint 已保存：{event['checkpoint_path']}"
    )


def idempotency_key(run_id: str, event: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{run_id}:{event['step']}:{event['checkpoint_path']}".encode()
    ).hexdigest()
    return f"state41-{event['step']}-{digest[:24]}"


def lark_command(
    *, user_id: str, identity: str, run_id: str, event: dict[str, Any], dry_run: bool
) -> list[str]:
    command = [
        "lark-cli",
        "im",
        "+messages-send",
        "--user-id",
        user_id,
        "--as",
        identity,
        "--text",
        message_text(run_id, event),
        "--idempotency-key",
        idempotency_key(run_id, event),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _new_ledger(run_id: str, user_id: str, identity: str) -> dict[str, Any]:
    return {
        "contract": LEDGER_CONTRACT,
        "run_id": run_id,
        "recipient_user_id": user_id,
        "identity": identity,
        "sent": {},
    }


def load_ledger(path: Path, *, run_id: str, user_id: str, identity: str) -> dict[str, Any]:
    if not path.exists():
        return _new_ledger(run_id, user_id, identity)
    ledger = json.loads(path.read_text())
    expected = {
        "contract": LEDGER_CONTRACT,
        "run_id": run_id,
        "recipient_user_id": user_id,
        "identity": identity,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise ValueError(f"notification ledger mismatch {key}: {ledger.get(key)!r}")
    if not isinstance(ledger.get("sent"), dict):
        raise ValueError("notification ledger sent field is not a mapping")
    return ledger


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incoming-{os.getpid()}")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def send_pending(
    events: Sequence[dict[str, Any]],
    ledger: dict[str, Any],
    *,
    user_id: str,
    identity: str,
    run_id: str,
    emit_only: bool,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    on_sent: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    sent_count = 0
    for event in events:
        step_key = str(event["step"])
        if step_key in ledger["sent"]:
            continue
        command = lark_command(
            user_id=user_id,
            identity=identity,
            run_id=run_id,
            event=event,
            dry_run=emit_only,
        )
        if emit_only:
            print(json.dumps({"would_send": command, "event": event}, ensure_ascii=False))
            continue
        result = run_command(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
                "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
            },
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Feishu send failed for step {event['step']}: {result.stderr.strip()}"
            )
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Feishu send returned invalid JSON for step {event['step']}"
            ) from error
        message_id = response.get("message_id")
        if not isinstance(message_id, str) or not message_id.startswith("om_"):
            raise RuntimeError(
                f"Feishu send lacks message_id for step {event['step']}: {response!r}"
            )
        ledger["sent"][step_key] = {
            "loss": float(event["loss"]),
            "checkpoint_path": event["checkpoint_path"],
            "message_id": message_id,
            "idempotency_key": idempotency_key(run_id, event),
            "sent_unix_seconds": time.time(),
        }
        sent_count += 1
        if on_sent is not None:
            on_sent(ledger)
    return sent_count


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if not args.user_id.startswith("ou_"):
        raise ValueError("--user-id must be a Feishu open_id beginning with ou_")
    args.ledger_json.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.ledger_json.with_suffix(args.ledger_json.suffix + ".lock")
    with lock_path.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another notifier owns {lock_path}") from error
        while True:
            try:
                text = _read_event_text(args.events_jsonl, args.ssh_host)
            except FileNotFoundError:
                if args.watch:
                    time.sleep(args.poll_seconds)
                    continue
                raise
            if text and not text.endswith("\n"):
                if args.watch:
                    time.sleep(args.poll_seconds)
                    continue
                raise ValueError("checkpoint event stream ends with a partial line")
            events = parse_events(text)
            ledger = load_ledger(
                args.ledger_json,
                run_id=args.run_id,
                user_id=args.user_id,
                identity=args.identity,
            )
            sent = send_pending(
                events,
                ledger,
                user_id=args.user_id,
                identity=args.identity,
                run_id=args.run_id,
                emit_only=args.emit_only,
                on_sent=(
                    None
                    if args.emit_only
                    else lambda current: save_ledger(args.ledger_json, current)
                ),
            )
            if args.emit_only or not args.watch:
                return 0
            if "100000" in ledger["sent"]:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
