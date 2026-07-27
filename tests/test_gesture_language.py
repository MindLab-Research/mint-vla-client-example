from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.gesture_language import (
    DEFAULT_GESTURE_INDEX_PATH,
    GestureIndex,
    format_gesture_prompt,
)


def _payload(entries: list[dict]) -> dict:
    return {
        "version": "1.0",
        "dataset_name": "new_all_generated_mano.lance",
        "entries": entries,
    }


def _entry(row: int, *, gesture: str = "02", seed: str = "seed-a") -> dict:
    return {
        "row_index": row,
        "uuid": f"uuid-{row}",
        "seed_uuid": seed,
        "object_type": "cube1",
        "gesture": gesture,
        "action_id": gesture,
        "sequence_id": "001",
        "trajectory_name": f"cube1_{gesture}_001",
        "total_frames": 10 + row,
    }


class GestureLanguageTests(unittest.TestCase):
    def test_canonical_index_contract(self) -> None:
        index = GestureIndex.load(DEFAULT_GESTURE_INDEX_PATH)
        self.assertEqual(len(index), 7539)
        self.assertEqual(
            index.sha256,
            "ec847b5dc3fa5f59e03849bec71e1eb5d2d8557ad0addfa2b52feba15ba0580f",
        )
        expected = {
            656: ("02", "08affaf6-d692-49e7-b254-f7961a7f2015"),
            995: ("04", "85c2ea71-6f17-4c7b-8497-46a2f85be7a3"),
            1155: ("09", "d5c73ea1-edc5-56bd-a492-0f1736e9a642"),
            1303: ("10", "6818f39d-0efd-4cf6-86f5-192185c9ff52"),
        }
        for row, (gesture, seed_uuid) in expected.items():
            record = index._records[row]
            self.assertEqual(
                (record.object_type, record.gesture, record.seed_uuid),
                ("cube1", gesture, seed_uuid),
            )

    def _load(self, payload: dict) -> GestureIndex:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "index.json"
        path.write_text(json.dumps(payload))
        return GestureIndex.load(path)

    def test_prompt_uses_canonical_gesture(self) -> None:
        self.assertEqual(
            format_gesture_prompt("  pick up the cube1  ", "02"),
            "pick up the cube1 using gesture 02",
        )

    def test_lookup_fails_closed_on_lance_mismatch(self) -> None:
        index = self._load(_payload([_entry(0)]))
        record = index.record_for(
            0,
            uuid="uuid-0",
            seed_uuid="seed-a",
            object_type="cube1",
            total_frames=10,
        )
        self.assertEqual(record.gesture, "02")
        for kwargs in (
            {"uuid": "wrong", "seed_uuid": "seed-a", "object_type": "cube1"},
            {"uuid": "uuid-0", "seed_uuid": "wrong", "object_type": "cube1"},
            {"uuid": "uuid-0", "seed_uuid": "seed-a", "object_type": "banana"},
        ):
            with self.assertRaisesRegex(ValueError, "does not align"):
                index.record_for(0, **kwargs)

    def test_index_rejects_noncontiguous_rows_and_gesture_conflicts(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self._load(_payload([_entry(1)]))
        with self.assertRaisesRegex(ValueError, "aliases gestures"):
            self._load(
                _payload([
                    _entry(0, gesture="02", seed="same"),
                    _entry(1, gesture="04", seed="same"),
                ])
            )

    def test_index_rejects_invalid_gesture_and_action_disagreement(self) -> None:
        invalid = _entry(0, gesture="2")
        with self.assertRaisesRegex(ValueError, "invalid gesture"):
            self._load(_payload([invalid]))
        mismatch = _entry(0)
        mismatch["action_id"] = "04"
        with self.assertRaisesRegex(ValueError, "gesture/action_id mismatch"):
            self._load(_payload([mismatch]))


if __name__ == "__main__":
    unittest.main()
