#!/usr/bin/env python3
"""Migrate persistent checkpoints from legacy PFS roots into the configured persistent root.

Persistent checkpoint cutover is fail-closed: a real run must remove source trees after
successful copy so runtime resolution cannot silently rely on legacy PFS roots.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from tinker_server import checkpoints


def _iter_checkpoint_dirs(root: str) -> list[Path]:
    base = Path(root)
    if not base.is_dir():
        return []
    out: list[Path] = []
    for owner in base.iterdir():
        if not owner.is_dir():
            continue
        for child in owner.iterdir():
            if not child.is_dir():
                continue
            grandchildren = [g for g in child.iterdir() if g.is_dir()]
            if (child / "metadata.json").exists():
                out.append(child)
            for grandchild in grandchildren:
                if (grandchild / "metadata.json").exists():
                    out.append(grandchild)
    return out


def _is_persistent_checkpoint(path: Path) -> bool:
    if checkpoints.is_ephemeral_checkpoint_name(path.name):
        return False
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}
        if metadata.get("checkpoint_type") in {"training", "sampler"}:
            return True
    return checkpoints.checkpoint_has_lora_weights(str(path)) or checkpoints.checkpoint_has_optimizer_state(str(path))


def _relative_dest_path(src_root: str, checkpoint_dir: Path) -> Path:
    return Path(os.path.relpath(checkpoint_dir, src_root))


def _copytree_with_retries(src: Path, dst: Path, attempts: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            shutil.copytree(src, dst)
            return
        except Exception as e:
            last_error = e
            shutil.rmtree(dst, ignore_errors=True)
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    if last_error is not None:
        raise last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", default=[], help="Legacy root to scan; may be repeated")
    parser.add_argument("--dest-root", default=checkpoints.get_persistent_checkpoints_dir())
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.delete_source:
        raise SystemExit("Real migration requires --delete-source; use --dry-run for inventory only.")

    source_roots = args.source_root or checkpoints.get_legacy_checkpoint_dirs()
    source_roots = [root for root in source_roots if os.path.realpath(root) != os.path.realpath(args.dest_root)]

    copied = 0
    skipped = 0
    vanished = 0
    failed = 0
    deleted = 0
    for src_root in source_roots:
        for checkpoint_dir in _iter_checkpoint_dirs(src_root):
            if not _is_persistent_checkpoint(checkpoint_dir):
                skipped += 1
                print(f"skip_nonpersistent {checkpoint_dir}")
                continue

            rel = _relative_dest_path(src_root, checkpoint_dir)
            dest_dir = Path(args.dest_root) / rel
            print(f"migrate {checkpoint_dir} -> {dest_dir}")
            if args.dry_run:
                continue

            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            try:
                _copytree_with_retries(checkpoint_dir, dest_dir)
            except FileNotFoundError:
                vanished += 1
                shutil.rmtree(dest_dir, ignore_errors=True)
                print(f"skip_vanished {checkpoint_dir}")
                continue
            except shutil.Error as e:
                missing_src = True
                for err in e.args[0]:
                    src_path = str(err[0])
                    if src_path.startswith(str(checkpoint_dir)) and not os.path.exists(src_path):
                        continue
                    missing_src = False
                    break
                if missing_src:
                    vanished += 1
                    shutil.rmtree(dest_dir, ignore_errors=True)
                    print(f"skip_vanished {checkpoint_dir}")
                    continue
                failed += 1
                shutil.rmtree(dest_dir, ignore_errors=True)
                print(f"copy_failed {checkpoint_dir}: {e}")
                continue
            except OSError as e:
                failed += 1
                shutil.rmtree(dest_dir, ignore_errors=True)
                print(f"copy_failed {checkpoint_dir}: {e}")
                continue
            copied += 1

            if args.delete_source:
                shutil.rmtree(checkpoint_dir)
                deleted += 1

    print(
        json.dumps(
            {
                "source_roots": source_roots,
                "dest_root": args.dest_root,
                "copied": copied,
                "deleted": deleted,
                "skipped": skipped,
                "vanished": vanished,
                "failed": failed,
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
