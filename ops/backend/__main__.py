from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .config import OpsBackendConfig
from .main import create_app
from .service import DirectMintOpsService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mint ops backend")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address")
    parser.add_argument("--backend-port", type=int, default=8787, help="Backend port")
    parser.add_argument("--mint-base-url", default=None, help="Mint API base URL, e.g. http://127.0.0.1:18000")
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray address. On a Ray driver node, omit this flag or set auto (default: auto).",
    )
    parser.add_argument("--api-key", default=None, help="Mint admin API key")
    parser.add_argument("--timeout-s", type=float, default=None, help="Mint HTTP timeout")
    parser.add_argument("--include-removed-pg", action="store_true", help="Include removed placement groups in state payload")
    parser.add_argument("--reload", action="store_true", help="Run uvicorn with reload")
    return parser


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    config = OpsBackendConfig.from_repo_root(repo_root)
    args = build_parser().parse_args()

    if args.mint_base_url is not None:
        config.mint_base_url = str(args.mint_base_url).rstrip("/")
    if args.ray_address is not None:
        config.ray_address = str(args.ray_address)
    if args.api_key is not None:
        config.api_key = args.api_key
    if args.timeout_s is not None:
        config.timeout_s = float(args.timeout_s)
    if args.include_removed_pg:
        config.include_removed_pg = True

    app = create_app(config=config, service=DirectMintOpsService(config))
    uvicorn.run(app, host=args.bind, port=int(args.backend_port), reload=bool(args.reload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
