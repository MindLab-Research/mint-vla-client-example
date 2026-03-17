from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _read_cors_origins() -> tuple[str, ...]:
    raw = (os.getenv("MINT_OPS_CORS_ORIGINS") or "").strip()
    if not raw:
        return ("http://127.0.0.1:5173", "http://localhost:5173")
    if raw == "*":
        return ("*",)
    origins = tuple(part.strip() for part in raw.split(",") if part.strip())
    return origins or ("http://127.0.0.1:5173", "http://localhost:5173")


@dataclass(slots=True)
class OpsBackendConfig:
    repo_root: Path
    mint_base_url: str
    ray_address: str
    api_key: str | None = None
    timeout_s: float = 20.0
    include_removed_pg: bool = False
    frontend_dist: Path | None = None
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "OpsBackendConfig":
        mint_base_url = (
            os.getenv("MINT_OPS_MINT_BASE_URL")
            or os.getenv("MINT_BASE_URL")
            or os.getenv("TINKER_BASE_URL")
            or ""
        ).strip()
        ray_address = (
            os.getenv("MINT_OPS_RAY_ADDRESS")
            or os.getenv("RAY_ADDRESS")
            or "auto"
        ).strip()
        api_key = (
            os.getenv("MINT_OPS_API_KEY")
            or os.getenv("MINT_API_KEY")
            or os.getenv("TINKER_API_KEY")
            or None
        )
        return cls(
            repo_root=repo_root,
            mint_base_url=mint_base_url.rstrip("/"),
            ray_address=ray_address,
            api_key=api_key,
            timeout_s=float(os.getenv("MINT_OPS_TIMEOUT_S", "20.0")),
            include_removed_pg=(os.getenv("MINT_OPS_INCLUDE_REMOVED_PG", "0") == "1"),
            frontend_dist=repo_root / "ops" / "frontend" / "dist",
            cors_origins=_read_cors_origins(),
        )

    def validate_runtime(self) -> None:
        if not self.mint_base_url:
            raise ValueError("mint_base_url is required")
        if not self.ray_address:
            raise ValueError("ray_address is required")

    @property
    def is_ready(self) -> bool:
        return bool(self.mint_base_url and self.ray_address)
