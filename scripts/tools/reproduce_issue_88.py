import io
import json
import sys
import tarfile
import tempfile
import types
import importlib.machinery
from pathlib import Path


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _install_fastapi_stub() -> None:
    try:
        import fastapi  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    from starlette.background import BackgroundTasks
    from starlette.datastructures import UploadFile
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse

    fastapi = types.ModuleType("fastapi")
    fastapi.__spec__ = importlib.machinery.ModuleSpec("fastapi", loader=None)

    class APIRouter:
        def post(self, *_args, **_kwargs):
            def deco(fn):
                return fn

            return deco

        def get(self, *_args, **_kwargs):
            def deco(fn):
                return fn

            return deco

        def delete(self, *_args, **_kwargs):
            def deco(fn):
                return fn

            return deco

    def File(*_args, **_kwargs):
        return None

    fastapi.APIRouter = APIRouter  # type: ignore[attr-defined]
    fastapi.BackgroundTasks = BackgroundTasks  # type: ignore[attr-defined]
    fastapi.File = File  # type: ignore[attr-defined]
    fastapi.HTTPException = HTTPException  # type: ignore[attr-defined]
    fastapi.Request = Request  # type: ignore[attr-defined]
    fastapi.Response = Response  # type: ignore[attr-defined]
    fastapi.UploadFile = UploadFile  # type: ignore[attr-defined]

    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.__spec__ = importlib.machinery.ModuleSpec("fastapi.responses", loader=None)
    fastapi_responses.StreamingResponse = StreamingResponse  # type: ignore[attr-defined]

    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.responses"] = fastapi_responses


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    _install_fastapi_stub()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Ensure weights.py picks up local temp checkpoints dir at import time.
        import os

        os.environ["MINT_CHECKPOINT_DIR"] = str(tmp_path)

        archive_path = tmp_path / "ckpt.tar.gz"
        root = "ckpt_original"
        payload = {
            f"{root}/adapter_model.safetensors": b"dummy-lora",
            f"{root}/optimizer.pt": b"dummy-optimizer",
            f"{root}/training_meta.json": json.dumps({"current_step": 7}).encode("utf-8"),
        }

        with tarfile.open(archive_path, "w:gz") as tf:
            for name, data in payload.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        from starlette.datastructures import UploadFile  # noqa: E402
        from starlette.exceptions import HTTPException  # noqa: E402
        import asyncio  # noqa: E402

        from mint_server.routes.weights import CHECKPOINTS_DIR, upload_checkpoint_archive  # noqa: E402

        dummy_req = types.SimpleNamespace(state=types.SimpleNamespace(user_data=None))
        upload = UploadFile(file=open(archive_path, "rb"), filename="ckpt.tar.gz")
        try:
            resp = asyncio.run(upload_checkpoint_archive(http_request=dummy_req, file=upload))
        finally:
            try:
                upload.file.close()
            except Exception:
                pass

        if not getattr(resp, "checkpoint_id", ""):
            return _fail(f"upload response missing checkpoint_id: {resp!r}")
        if resp.path != resp.checkpoint_id:
            return _fail(f"upload response path={resp.path!r} expected {resp.checkpoint_id!r}")

        final_dir = Path(CHECKPOINTS_DIR) / "anonymous" / resp.checkpoint_id
        if not final_dir.exists():
            return _fail(f"checkpoint dir not created: {final_dir}")
        if not (final_dir / "adapter_model.safetensors").exists():
            return _fail("missing adapter_model.safetensors after upload")
        if not (final_dir / "optimizer.pt").exists():
            return _fail("missing optimizer.pt after upload")
        if not (final_dir / "training_meta.json").exists():
            return _fail("missing training_meta.json after upload")
        if (final_dir / root).exists():
            return _fail("archive root dir not stripped during upload")
        if not (final_dir / "metadata.json").exists():
            return _fail("missing metadata.json after upload")

        # Invalid archive: path traversal should return HTTP 400.
        evil_path = tmp_path / "evil.tar.gz"
        with tarfile.open(evil_path, "w:gz") as tf:
            data = b"evil"
            info = tarfile.TarInfo(name="ckpt_x/../../evil.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        upload2 = UploadFile(file=open(evil_path, "rb"), filename="evil.tar.gz")
        try:
            try:
                asyncio.run(upload_checkpoint_archive(http_request=dummy_req, file=upload2))
                return _fail("expected HTTPException for unsafe archive")
            except HTTPException as e:
                if int(getattr(e, "status_code", 0)) != 400:
                    return _fail(f"unsafe archive status_code={getattr(e, 'status_code', None)!r} expected 400")
        finally:
            try:
                upload2.file.close()
            except Exception:
                pass

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
