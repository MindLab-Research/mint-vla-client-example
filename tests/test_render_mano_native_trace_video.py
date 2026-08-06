from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "client_render_mano_native_trace_video",
    Path(__file__).parents[1] / "tools/render_mano_native_trace_video.py",
)
assert SPEC is not None and SPEC.loader is not None
video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video)


def test_probe_fallback_verifies_encoded_h264_population(tmp_path, monkeypatch):
    import imageio

    path = tmp_path / "probe.mp4"
    writer = imageio.get_writer(path, fps=50, macro_block_size=1)
    for value in (0, 40, 80):
        writer.append_data(np.full((32, 64, 3), value, dtype=np.uint8))
    writer.close()
    monkeypatch.setattr(video.shutil, "which", lambda _name: None)

    probe = video._probe_video(path)
    video._validate_probe(probe, width=64, height=32, fps=50, frames=3)

    assert probe["probe_backend"] == "imageio-ffmpeg"
    assert probe["streams"][0]["codec_name"] == "h264"
    assert probe["streams"][0]["pix_fmt"] == "yuv420p"


def test_video_contract_rejects_wrong_frame_count():
    probe = {
        "streams": [{
            "codec_name": "h264", "pix_fmt": "yuv420p", "width": 64,
            "height": 32, "r_frame_rate": "50/1", "nb_frames": "2",
        }],
        "format": {"duration": "0.04", "size": "100"},
    }
    with pytest.raises(RuntimeError, match="frame contract"):
        video._validate_probe(probe, width=64, height=32, fps=50, frames=3)
