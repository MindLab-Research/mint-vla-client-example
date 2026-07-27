from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import contact_windows as cw


class _Field:
    def __init__(self, name: str) -> None:
        self.name = name


class _Schema:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _Table:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def to_pylist(self) -> list[dict]:
        return self._rows


class _Dataset:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        names: set[str] = set()
        for row in rows:
            names.update(row)
        self.schema = _Schema(sorted(names))
        self.take_calls: list[list[int]] = []

    def count_rows(self) -> int:
        return len(self.rows)

    def take(self, indices: list[int], *, columns: list[str]) -> _Table:
        self.take_calls.append(list(indices))
        return _Table([{key: self.rows[index].get(key) for key in columns} for index in indices])


def _row(total: int, first: int | None, last: int | None, *, object_name: str = "cube1") -> dict:
    contact: list[list[dict]] = [[] for _ in range(total)]
    if first is not None:
        contact[first].append({"object_name": object_name, "joint_name": "thumb3"})
    if last is not None and last != first:
        contact[last].append({"object_name": object_name, "joint_name": "index3"})
    return {
        "trajectory_metadata": {"object_names": [object_name]},
        "episode_metadata": {"total_frames": total},
        "contact": contact,
    }


class ContactWindowTests(unittest.TestCase):
    def test_contact_window_is_inclusive_and_clamped(self) -> None:
        row = _row(715, 240, 528)
        window = cw.window_from_row(row, row_index=656)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual((window.first_contact_frame, window.last_contact_frame), (240, 528))
        self.assertEqual((window.start_frame, window.end_frame), (140, 628))
        self.assertEqual(window.frame_count, 489)
        self.assertEqual(window.status, "contact_window")

        left = cw.window_from_row(_row(150, 20, 100), row_index=1)
        assert left is not None
        self.assertEqual((left.start_frame, left.end_frame), (0, 149))

    def test_only_target_object_contact_counts(self) -> None:
        row = _row(300, None, None)
        row["contact"][10] = [{"object_name": "other"}]
        row["contact"][50] = [{"object_name": "cube1"}]
        row["contact"][70] = [{"object_name": "other"}, {"object_name": "cube1"}]
        window = cw.window_from_row(row, row_index=2, context_frames=5)
        assert window is not None
        self.assertEqual((window.first_contact_frame, window.last_contact_frame), (50, 70))
        self.assertEqual((window.start_frame, window.end_frame), (45, 75))
        self.assertEqual(window.contact_frame_count, 2)
        self.assertEqual(window.contact_record_frame_count, 3)

    def test_missing_contact_policy_is_explicit(self) -> None:
        row = {
            "trajectory_metadata": {"object_names": ["cube1"]},
            "episode_metadata": {"total_frames": 80},
        }
        full = cw.window_from_row(row, row_index=3, missing_policy="full")
        assert full is not None
        self.assertEqual((full.start_frame, full.end_frame), (0, 79))
        self.assertEqual(full.status, "full_fallback:missing_contact_column")
        self.assertIsNone(cw.window_from_row(row, row_index=3, missing_policy="skip"))
        with self.assertRaisesRegex(ValueError, "no valid contact interval"):
            cw.window_from_row(row, row_index=3, missing_policy="error")

    def test_full_override_does_not_depend_on_contact(self) -> None:
        row = _row(715, 240, 528)
        window = cw.select_window(
            row, row_index=656, total_frames=715, mode="full"
        )
        assert window is not None
        self.assertEqual((window.start_frame, window.end_frame), (0, 714))
        self.assertEqual(window.status, "full_fallback:explicit_full_override")

    def test_manifest_scan_is_incremental_and_round_trips(self) -> None:
        dataset = _Dataset([_row(715, 240, 528), _row(656, 232, 520)])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "contact.json"
            first = cw.load_or_build_windows(
                dataset,
                "/fake/data.lance",
                [0],
                manifest_path=manifest,
                batch_rows=1,
            )
            self.assertEqual(first[0]["start_frame"], 140)
            self.assertEqual(dataset.take_calls, [[0]])
            raw = json.loads(manifest.read_text())
            self.assertEqual(sorted(raw["windows"]), ["0"])

            both = cw.load_or_build_windows(
                dataset,
                "/fake/data.lance",
                [0, 1],
                manifest_path=manifest,
                batch_rows=1,
            )
            self.assertEqual(dataset.take_calls, [[0], [1]])
            self.assertEqual((both[1]["start_frame"], both[1]["end_frame"]), (132, 620))
            loaded, entries = cw.load_manifest(manifest)
            self.assertEqual(loaded["context_frames"], 100)
            self.assertEqual(sorted(entries), [0, 1])

    def test_manifest_contract_mismatch_fails_without_mutation(self) -> None:
        dataset = _Dataset([
            {
                "trajectory_metadata": {"object_names": ["cube1"]},
                "episode_metadata": {"total_frames": 10},
                "contact": [[] for _ in range(10)],
            }
        ])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "contact.json"
            cw.load_or_build_windows(dataset, "/fake/data.lance", [0], manifest_path=manifest, missing_policy="full")
            original = manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, "distinct manifest path"):
                cw.load_or_build_windows(dataset, "/fake/data.lance", [0], manifest_path=manifest, missing_policy="skip")
            self.assertEqual(manifest.read_bytes(), original)

    def test_clamp_manifest_window_to_loaded_arrays(self) -> None:
        window = cw.window_from_row(_row(715, 240, 528), row_index=656)
        assert window is not None
        clamped = cw.clamp_window(window, 600)
        self.assertEqual((clamped.start_frame, clamped.end_frame), (140, 599))
        empty = cw.clamp_window(window, 0)
        self.assertEqual((empty.start_frame, empty.end_frame, empty.frame_count), (0, -1, 0))
        self.assertEqual(empty.status, "empty_trajectory")


if __name__ == "__main__":
    unittest.main()
