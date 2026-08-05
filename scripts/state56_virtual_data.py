"""Authenticated join between the State41 source Lance and State56 sidecar."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import lance
import numpy as np

from scripts import mano_state56_contract as C

SIDECAR_CONTRACT = "mano_state56_native28_virtual_sidecar_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class State56SidecarStore:
    """Read-only, fail-closed State56 state join keyed by source release row."""

    def __init__(
        self,
        sidecar_path: Path,
        *,
        verification_path: Path,
        expected_verification_sha256: str,
        source_dataset: Path,
    ) -> None:
        self.path = Path(sidecar_path).expanduser().resolve()
        self.verification_path = Path(verification_path).expanduser().resolve()
        self.source_dataset = Path(source_dataset).expanduser().resolve()
        expected = str(expected_verification_sha256).lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError("State56 sidecar verification SHA must be64 hexadecimal characters")
        actual = sha256(self.verification_path)
        if actual != expected:
            raise ValueError(f"State56 sidecar verification SHA mismatch: {actual} != {expected}")
        verification = json.loads(self.verification_path.read_text(encoding="utf-8"))
        required = {
            "contract": SIDECAR_CONTRACT,
            "status": "passed",
            "path": str(self.path),
            "rows": 4856,
            "train_rows": 4613,
            "validation_rows": 243,
            "held_out_rows": 0,
            "frames": 2695132,
            "state_dim": C.STATE_DIM,
            "action_dim": C.ACTION_DIM,
        }
        for key, value in required.items():
            if verification.get(key) != value:
                raise ValueError(
                    f"State56 sidecar verification {key!r} mismatch: "
                    f"expected {value!r}, got {verification.get(key)!r}"
                )
        self.verification = verification
        self.verification_sha256 = actual
        self.release_sha256 = actual
        self.plan_sha256 = str(verification["plan_sha256"])
        self.dataset = lance.dataset(str(self.path), version=int(verification["lance_version"]))
        light = self.dataset.to_table(columns=["index", "window", "provenance"]).to_pylist()
        if len(light) != 4856:
            raise ValueError(f"State56 sidecar row count mismatch: {len(light)}")
        positions: dict[int, int] = {}
        split_counts = {"train": 0, "validation": 0}
        frames = 0
        source_version: int | None = None
        for position, row in enumerate(light):
            index = row["index"]
            source_row = int(index["release_row_index"])
            if source_row in positions:
                raise ValueError(f"duplicate State56 source release row {source_row}")
            split = index["split"]
            if split not in split_counts:
                raise ValueError(f"unknown State56 split {split!r}")
            split_counts[split] += 1
            window = row["window"]
            if (
                int(window["start_frame"]) != 0
                or int(window["frame_count"]) != int(window["source_total_frames"])
                or int(window["end_frame"]) != int(window["source_total_frames"]) - 1
            ):
                raise ValueError(f"State56 Scheme-A sidecar row {source_row} is not a full contact window")
            provenance = row["provenance"]
            if Path(provenance["source_dataset"]).resolve() != self.source_dataset:
                raise ValueError(f"State56 sidecar source dataset mismatch row {source_row}")
            if provenance["plan_sha256"] != self.plan_sha256:
                raise ValueError(f"State56 sidecar plan mismatch row {source_row}")
            row_source_version = int(provenance["source_dataset_version"])
            if source_version is None:
                source_version = row_source_version
            elif row_source_version != source_version:
                raise ValueError("State56 sidecar source version varies by row")
            positions[source_row] = position
            frames += int(window["frame_count"])
        if split_counts != {"train": 4613, "validation": 243} or frames != 2695132:
            raise ValueError(f"State56 sidecar population mismatch: {split_counts}, frames={frames}")
        self.source_dataset_version = int(source_version)
        self._positions = positions

    def has_source_row(self, source_row: int) -> bool:
        return int(source_row) in self._positions

    def load(self, source_row: int, *, expected_uuid: str, expected_source_payload_sha256: str) -> dict[str, Any]:
        try:
            position = self._positions[int(source_row)]
        except KeyError as exc:
            raise ValueError(f"source row {source_row} is outside the Grade-A State56 sidecar") from exc
        row = self.dataset.take(
            [position], columns=["index", "window", "state", "state_sha256", "provenance"]
        ).to_pylist()[0]
        if row["index"]["release_row_index"] != int(source_row) or row["index"]["uuid"] != expected_uuid:
            raise ValueError(f"State56 sidecar UUID/source-row mismatch at {source_row}")
        if row["provenance"]["source_row_payload_sha256"] != expected_source_payload_sha256:
            raise ValueError(f"State56 sidecar source payload mismatch at {source_row}")
        state = np.asarray(row["state"], dtype=np.float32)
        expected_shape = (int(row["window"]["source_total_frames"]), C.STATE_DIM)
        if state.shape != expected_shape or not np.all(np.isfinite(state)):
            raise ValueError(f"State56 sidecar state is invalid at {source_row}: {state.shape}")
        state_sha = hashlib.sha256(np.ascontiguousarray(state, dtype="<f4").tobytes()).hexdigest()
        if state_sha != row["state_sha256"]:
            raise ValueError(f"State56 sidecar state SHA mismatch at {source_row}")
        return {**row, "state": state}
