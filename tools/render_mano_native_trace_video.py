#!/usr/bin/env python3
"""Render a qualified native trace with the unchanged Client visual contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import imageio
import numpy as np
from PIL import Image

from scripts.eval import mano_action_support
from scripts.eval import manorl_native_physics as physics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


def _probe_video(path: Path) -> dict:
    executable = shutil.which("ffprobe")
    if executable:
        command = [
            executable, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration,size", "-of", "json", str(path),
        ]
        payload = json.loads(subprocess.check_output(command, text=True))
        payload["probe_backend"] = "ffprobe"
    else:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            metadata = next(reader)
        finally:
            reader.close()
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        width, height = metadata["size"]
        fps = float(metadata["fps"])
        payload = {
            "streams": [{
                "codec_name": metadata["codec"],
                "pix_fmt": metadata["pix_fmt"].split("(", 1)[0],
                "width": width, "height": height,
                "r_frame_rate": f"{int(round(fps))}/1",
                "nb_frames": str(frames), "duration": str(seconds),
            }],
            "format": {"duration": str(seconds), "size": str(path.stat().st_size)},
            "probe_backend": "imageio-ffmpeg",
            "ffmpeg_version": metadata["ffmpeg_version"],
        }
    if len(payload.get("streams", [])) != 1:
        raise RuntimeError(f"video probe did not find exactly one video stream: {payload}")
    return payload


def _validate_probe(probe: dict, *, width: int, height: int, fps: int, frames: int) -> None:
    stream = probe["streams"][0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"unexpected video codec/pixel format: {stream}")
    if int(stream.get("width", -1)) != width or int(stream.get("height", -1)) != height:
        raise RuntimeError(f"unexpected video size: {stream}")
    numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
    if numerator / denominator != fps or int(stream.get("nb_frames", -1)) != frames:
        raise RuntimeError(f"unexpected video frame contract: {stream}")
    duration = float(probe["format"]["duration"])
    if abs(duration - frames / fps) > 1e-3:
        raise RuntimeError(f"unexpected video duration {duration} != {frames / fps}")


def render(
    *,
    trace_path: Path,
    report_path: Path,
    output_path: Path,
    cover_path: Path,
    manifest_path: Path,
    object_name: str,
    camera_name: str,
    head_camera_preset: str,
    width: int,
    height: int,
    fps: int,
    cover_frame: int,
) -> dict:
    import mujoco

    report = json.loads(report_path.read_text())
    if report.get("status") != "ok" or report.get("object") != object_name:
        raise ValueError("trace report is not a qualified replay for the requested object")
    if report.get("trace_sha256") != _sha256(trace_path):
        raise ValueError("trace/report SHA256 mismatch")
    with np.load(trace_path) as trace:
        qpos = np.asarray(trace["simulated_full_qpos"], dtype=np.float64)
        qvel = np.asarray(trace["simulated_full_qvel"], dtype=np.float64)
        simulated_time = np.asarray(trace["simulated_time"], dtype=np.float64)
    frames = len(qpos)
    if qpos.shape != (frames, 35) or qvel.shape != (frames, 34) or simulated_time.shape != (frames,):
        raise ValueError(f"invalid native trace shapes: {qpos.shape} {qvel.shape} {simulated_time.shape}")
    camera = {
        "head": mano_action_support.HEAD_CAMERA_NAME,
        "wrist": mano_action_support.WRIST_CAMERA_NAME,
    }[camera_name]
    _, model, data, renderer, *_ = physics.make_scene(
        object_name, width, height, physics=True, create_renderer=True,
        head_camera_preset=head_camera_preset,
    )
    if (model.nq, model.nv, model.nu) != (35, 34, 28):
        raise ValueError("visual model does not match the 28D native ABI")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise ValueError(f"missing camera {camera}")
    cover_frame = min(max(0, cover_frame), frames - 1)
    local_root = Path(os.environ.get("LOCAL_VIDEO_TMPDIR", "/tmp"))
    local_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mano28-video-", dir=local_root) as directory:
        directory = Path(directory)
        local_video = directory / "replay.mp4"
        local_cover = directory / "cover.png"
        writer = imageio.get_writer(local_video, fps=fps, macro_block_size=1)
        try:
            for index in range(frames):
                data.qpos[:] = qpos[index]
                data.qvel[:] = qvel[index]
                data.time = simulated_time[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                if frame.shape != (height, width, 3):
                    raise RuntimeError(f"unexpected render shape {frame.shape}")
                writer.append_data(frame)
                if index == cover_frame:
                    Image.fromarray(frame).save(local_cover)
        finally:
            writer.close()
            renderer.close()
        probe = _probe_video(local_video)
        _validate_probe(probe, width=width, height=height, fps=fps, frames=frames)
        _atomic_copy(local_video, output_path)
        _atomic_copy(local_cover, cover_path)
    manifest = {
        "contract": "mano_native_trace_video_legacy_visual_v1",
        "object": object_name,
        "row_index": int(report["row_index"]),
        "source_identity": report.get("source_identity"),
        "camera": camera_name,
        "camera_name": camera,
        "head_camera_preset": head_camera_preset,
        "camera_position": model.cam_pos[camera_id].tolist(),
        "camera_fovy": float(model.cam_fovy[camera_id]),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps,
        "playback_slowdown": (frames / fps) / float(simulated_time[-1]),
        "dynamics_steps_during_render": 0,
        "visual_physics_invariance": physics.visual_invariance(model),
        "source_trace": str(trace_path),
        "source_trace_sha256": _sha256(trace_path),
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "video": str(output_path),
        "video_sha256": _sha256(output_path),
        "cover": str(cover_path),
        "cover_frame": cover_frame,
        "cover_sha256": _sha256(cover_path),
        "ffprobe": probe,
        "renderer_script_sha256": _sha256(Path(__file__)),
        "camera_contract_source": str(Path(mano_action_support.__file__).resolve()),
        "camera_contract_source_sha256": _sha256(Path(mano_action_support.__file__).resolve()),
        "client_commit": os.environ.get("VLA_CLIENT_GIT_COMMIT", "unknown"),
        "manorl": physics.runtime_provenance(object_name),
    }
    temporary = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cover", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--object", default="cube1")
    parser.add_argument("--camera", choices=("head", "wrist"), default="head")
    parser.add_argument(
        "--head-camera-preset",
        choices=tuple(mano_action_support.HEAD_CAMERA_PRESETS),
        default=mano_action_support.DEFAULT_HEAD_CAMERA_PRESET,
        help="Head camera geometry; default is the current elevated 65-degree view",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--cover-frame", type=int, default=235)
    args = parser.parse_args()
    manifest = render(
        trace_path=Path(args.trace).resolve(), report_path=Path(args.report).resolve(),
        output_path=Path(args.output).resolve(), cover_path=Path(args.cover).resolve(),
        manifest_path=Path(args.manifest).resolve(), object_name=args.object,
        camera_name=args.camera, head_camera_preset=args.head_camera_preset,
        width=args.width, height=args.height,
        fps=args.fps, cover_frame=args.cover_frame,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
