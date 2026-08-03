from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from scripts.eval import build_mano_state46_release as release


pytestmark = pytest.mark.skipif(
    not (release.DEFAULT_QUALITY_ROOT / "global_grade_summary.json").is_file(),
    reason="native quality v2 artifacts unavailable",
)


def _entry(record):
    return {
        "filtered_row_index": int(record["row_index"]),
        "original_merged_row_index": int(record["original_merged_row_index"]),
        "row_uuid": record["row_uuid"],
        "seed_uuid": record["seed_uuid"],
        "object": record["object"],
        "gesture": record["gesture"],
        "grade": record["grade"],
        "frames": int(record["frames"]),
        "trace_npz": record["trace_npz"],
        "trace_sha256": record["trace_sha256"],
        "quality_report": str(
            release.DEFAULT_QUALITY_ROOT
            / "objects"
            / record["object"]
            / "records"
            / f"{int(record['row_index']):05d}.json"
        ),
    }


def test_contiguous_weighted_shards_preserve_global_order():
    records = [
        {"row_uuid": f"u{i}", "frames": frames, "object": "banana"}
        for i, frames in enumerate([10, 20, 40, 80, 10, 30, 50])
    ]
    shards = release.contiguous_weighted_shards(records, 3)
    assert [value["row_uuid"] for shard in shards for value in shard] == [
        value["row_uuid"] for value in records
    ]
    assert len(shards) == 3 and all(shards)


def test_quaternion_axis_angle_is_sign_invariant():
    q = np.asarray([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]])
    first = release.quaternion_wxyz_to_axis_angle(q)
    second = release.quaternion_wxyz_to_axis_angle(-q)
    np.testing.assert_allclose(first, second, atol=1e-7)
    np.testing.assert_allclose(first, [[0.0, 0.0, np.pi / 2]], atol=1e-7)


def test_one_row_release_uses_same_native_mjdata_and_schema(monkeypatch):
    import lance
    import mujoco
    import pyarrow as pa
    from PIL import Image

    monkeypatch.setattr(
        mujoco,
        "mj_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("release rendering must not step dynamics")
        ),
    )
    record = release.quality_records(release.DEFAULT_QUALITY_ROOT)[0]
    entry = _entry(record)
    dataset = lance.dataset(
        release.replay_quality.DEFAULT_DATASET,
        version=release.replay_quality.EXPECTED_DATASET_VERSION,
    )
    source = dataset.take(
        [entry["filtered_row_index"]], columns=release.SOURCE_COLUMNS
    ).to_pylist()[0]
    try:
        row = release.render_release_row(entry, source, plan_sha="a" * 64)
    finally:
        release.close_scenes()

    frames = entry["frames"]
    state = np.asarray(row["state"])
    actions = np.asarray(row["actions"])
    assert state.shape == (frames, 46)
    assert actions.shape == (frames, 32)
    np.testing.assert_array_equal(actions[:, 28:], 0.0)
    assert len(row["image"]) == len(row["wrist_image"]) == frames
    assert len(row["contact"]) == frames
    assert row["provenance"]["dynamics_steps_during_render"] == 0
    assert row["hands"][0]["hand_name"] == "right"
    assert len(row["row_payload_sha256"]) == 64
    for encoded in (row["image"][0], row["wrist_image"][0]):
        with Image.open(BytesIO(encoded)) as image:
            assert image.size == (640, 360)
            assert image.format == "JPEG"
    table = pa.Table.from_pylist([row], schema=release.release_schema())
    assert table.num_rows == 1
    assert table.schema.field("state").type.value_type.list_size == 46
    assert table.schema.field("actions").type.value_type.list_size == 32
