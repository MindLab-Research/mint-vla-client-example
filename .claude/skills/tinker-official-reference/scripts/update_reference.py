#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


COOKBOOK_RAW_REPO_DEFAULT = "https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/refs/heads/main"


def _fetch_text(url: str, timeout_s: float, retries: int, backoff_s: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "mint-tinker-official-reference-updater/1.0"})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            # Keep UTF-8; upstream may include non-ASCII.
            return data.decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last_err = e
            if attempt + 1 >= retries:
                break
            sleep_s = backoff_s * (2**attempt)
            print(f"warning: fetch failed for {url}; retrying in {sleep_s:.1f}s: {e}", file=sys.stderr)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update bundled Tinker official reference snapshot.")
    p.add_argument(
        "--cookbook-raw-repo-url",
        default=COOKBOOK_RAW_REPO_DEFAULT,
        help="GitHub raw repo URL (default: tinker-cookbook main branch)",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Network timeout in seconds (default: 30)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Fetch attempts per file (default: 4)",
    )
    p.add_argument(
        "--retry-backoff-s",
        type=float,
        default=1.0,
        help="Base backoff in seconds between retries (default: 1.0)",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "references" / "upstream"),
        help="Directory to write section files into (default: skill references/upstream/)",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    upstream_dir = Path(args.out_dir).resolve()

    cookbook_raw_repo_url = args.cookbook_raw_repo_url.rstrip("/")

    # Section-specific sources live in the tinker-cookbook repo. llms-full.txt explicitly
    # points users there; we keep the per-section files as a structured reference.
    docs_paths = [
        "docs/index.mdx",
        "docs/training-sampling.mdx",
        "docs/async.mdx",
        "docs/losses.mdx",
        "docs/save-load.mdx",
        "docs/download-weights.mdx",
        "docs/publish-weights.mdx",
        "docs/lora-primer.mdx",
        "docs/rendering.mdx",
        "docs/under-the-hood.mdx",
        "docs/dev-tips.mdx",
        "docs/evals.mdx",
        "docs/model-lineup.mdx",
        "docs/overview-building.mdx",
        "docs/compatible-apis/openai.mdx",
        "docs/completers.mdx",
        "docs/install.mdx",
        "docs/docs-outline.mdx",
        "docs/preferences.mdx",
        "docs/supervised-learning.mdx",
        "docs/rl.mdx",
        "docs/rl/rl-basic.mdx",
        "docs/rl/rl-envs.mdx",
        "docs/rl/rl-hyperparams.mdx",
        "docs/rl/rl-loops.mdx",
        "docs/rl/sequence-extension.mdx",
        "docs/preferences/dpo-guide.mdx",
        "docs/preferences/rlhf-example.mdx",
        "docs/supervised-learning/sl-basic.mdx",
        "docs/supervised-learning/sl-loop.mdx",
        "docs/supervised-learning/sl-hyperparams.mdx",
        "docs/supervised-learning/sweep-case-study.mdx",
        "docs/supervised-learning/prompt-distillation.mdx",
        "docs/api-reference/serviceclient.md",
        "docs/api-reference/trainingclient.md",
        "docs/api-reference/samplingclient.md",
        "docs/api-reference/restclient.md",
        "docs/api-reference/apifuture.md",
        "docs/api-reference/types.md",
        "docs/api-reference/exceptions.md",
    ]

    fetched: dict[str, str] = {}
    for rel in docs_paths:
        url = f"{cookbook_raw_repo_url}/{rel}"
        try:
            fetched[rel] = _fetch_text(url, args.timeout_s, args.retries, args.retry_backoff_s)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"error: failed to fetch {rel}: {e}", file=sys.stderr)
            return 2
        if len(fetched[rel]) < 200:
            print(f"error: fetched {rel} too small ({len(fetched[rel])} bytes)", file=sys.stderr)
            return 2

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = [
        "Tinker official reference (section files)\n",
        f"Updated: {stamp}\n",
        f"Source repo: {cookbook_raw_repo_url}\n",
        "\n",
        "These files are fetched from the official tinker-cookbook repository.\n",
        "Do not edit them manually; rerun update_reference.py.\n",
        "\n",
    ]
    _write_atomic(upstream_dir / "README.txt", "".join(meta))
    for rel, text in fetched.items():
        dst = upstream_dir / rel
        _write_atomic(dst, text.rstrip() + "\n")

    print(f"wrote section files under {upstream_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
