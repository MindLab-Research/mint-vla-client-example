from __future__ import annotations

import json
import logging
import math
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TrainingWorkerInputContract:
    """Input contract validation for dense `TrainingWorker.forward_backward`."""

    def __init__(
        self,
        *,
        base_model: Callable[[], str],
        vocab_size: Callable[[], int | None],
        request_id: Callable[[], str | None],
        record_span_event: Callable[[str], None] | Callable[..., None],
        record_training_incident: Callable[..., None],
    ) -> None:
        self._base_model = base_model
        self._vocab_size = vocab_size
        self._request_id = request_id
        self._record_span_event = record_span_event
        self._record_training_incident = record_training_incident

    @staticmethod
    def numeric_bounds(values: list[Any]) -> tuple[int | float | None, int | float | None]:
        numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if not numeric:
            return None, None
        return min(numeric), max(numeric)

    @staticmethod
    def invalid_token_positions(values: list[Any], *, vocab_size: int | None, limit: int = 8) -> list[int]:
        bad: list[int] = []
        for idx, value in enumerate(values):
            is_valid_int = isinstance(value, int) and not isinstance(value, bool)
            if not is_valid_int:
                bad.append(idx)
            elif int(value) < 0:
                bad.append(idx)
            elif vocab_size is not None and int(value) >= int(vocab_size):
                bad.append(idx)
            if len(bad) >= limit:
                break
        return bad

    @staticmethod
    def invalid_numeric_positions(values: list[Any], *, limit: int = 8) -> list[int]:
        bad: list[int] = []
        for idx, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                bad.append(idx)
            elif not math.isfinite(float(value)):
                bad.append(idx)
            if len(bad) >= limit:
                break
        return bad

    def validate_forward_backward(
        self,
        *,
        session_id: str | None,
        loss_fn: str,
        input_ids: list[Any],
        target_tokens: list[Any],
        weights: list[Any],
        old_logprobs: list[Any] | None = None,
        advantages: list[Any] | None = None,
    ) -> None:
        vocab_size = self._vocab_size()
        if self.invalid_token_positions(input_ids, vocab_size=vocab_size):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="input_ids_out_of_range",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if self.invalid_token_positions(target_tokens, vocab_size=vocab_size):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="target_tokens_out_of_range",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if self.invalid_numeric_positions(weights):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="weights_non_finite_or_non_numeric",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if len(target_tokens) != len(weights):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="target_weights_len_mismatch",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if len(target_tokens) != len(input_ids):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="target_seq_len_mismatch",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if old_logprobs is not None and old_logprobs and self.invalid_numeric_positions(old_logprobs):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="old_logprobs_non_finite_or_non_numeric",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if old_logprobs is not None and old_logprobs and len(old_logprobs) != len(target_tokens):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="old_logprobs_len_mismatch",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if advantages is not None and advantages and self.invalid_numeric_positions(advantages):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="advantages_non_finite_or_non_numeric",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )
        if advantages is not None and advantages and len(advantages) != len(target_tokens):
            self._raise_violation(
                session_id=session_id,
                loss_fn=loss_fn,
                reason="advantages_len_mismatch",
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
                old_logprobs=old_logprobs,
                advantages=advantages,
            )

    def _raise_violation(
        self,
        *,
        session_id: str | None,
        loss_fn: str,
        reason: str,
        input_ids: list[Any],
        target_tokens: list[Any],
        weights: list[Any],
        old_logprobs: list[Any] | None = None,
        advantages: list[Any] | None = None,
    ) -> None:
        vocab_size = self._vocab_size()
        input_min, input_max = self.numeric_bounds(input_ids)
        target_min, target_max = self.numeric_bounds(target_tokens)
        attrs = {
            "session_id": str(session_id or "-"),
            "loss_fn": str(loss_fn),
            "reason": str(reason),
            "vocab_size": -1 if vocab_size is None else int(vocab_size),
            "input_len": len(input_ids),
            "target_len": len(target_tokens),
            "weights_len": len(weights),
            "old_logprobs_len": len(old_logprobs or []),
            "advantages_len": len(advantages or []),
            "input_min": "none" if input_min is None else str(input_min),
            "input_max": "none" if input_max is None else str(input_max),
            "target_min": "none" if target_min is None else str(target_min),
            "target_max": "none" if target_max is None else str(target_max),
            "bad_input_positions": json.dumps(
                self.invalid_token_positions(input_ids, vocab_size=vocab_size),
                separators=(",", ":"),
            ),
            "bad_target_positions": json.dumps(
                self.invalid_token_positions(target_tokens, vocab_size=vocab_size),
                separators=(",", ":"),
            ),
            "bad_weight_positions": json.dumps(
                self.invalid_numeric_positions(weights),
                separators=(",", ":"),
            ),
            "bad_old_logprob_positions": json.dumps(
                self.invalid_numeric_positions(old_logprobs or []),
                separators=(",", ":"),
            ),
            "bad_advantage_positions": json.dumps(
                self.invalid_numeric_positions(advantages or []),
                separators=(",", ":"),
            ),
        }
        logger.error(
            "[TrainingWorker] dense_input_contract_violation %s",
            json.dumps(attrs, sort_keys=True, ensure_ascii=True),
        )
        self._record_span_event("mint.training_input_contract_violation", attributes=attrs)
        self._record_training_incident(
            kind="contract_violation",
            base_model=str(self._base_model() or "unknown"),
            backend="peft",
            op="forward_backward",
            status="error",
            failure_class="input_contract",
            request_id=str(self._request_id() or "") or None,
            session_id=None if session_id is None else str(session_id),
            detail=str(reason),
            context=attrs,
        )
        raise ValueError(
            "dense_input_contract_violation: "
            f"reason={reason} input_len={len(input_ids)} target_len={len(target_tokens)} weights_len={len(weights)}"
        )
