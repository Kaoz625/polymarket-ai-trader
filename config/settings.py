"""
Configuration for the Polymarket AI trading system.
All sensitive values loaded from environment variables via .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root (one level up from config/)
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    # ── API credentials ────────────────────────────────────────────────────
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    poly_api_key: str = field(
        default_factory=lambda: os.environ.get("POLY_API_KEY", "")
    )
    poly_api_secret: str = field(
        default_factory=lambda: os.environ.get("POLY_API_SECRET", "")
    )
    poly_api_passphrase: str = field(
        default_factory=lambda: os.environ.get("POLY_API_PASSPHRASE", "")
    )
    poly_private_key: str = field(
        default_factory=lambda: os.environ.get("POLY_PRIVATE_KEY", "")
    )

    # ── Exit / risk thresholds ─────────────────────────────────────────────
    # Exit when we have captured this fraction of the expected move
    exit_threshold: float = float(os.environ.get("EXIT_THRESHOLD", "0.85"))
    # Cut the position when it is down this fraction of entry price
    loss_cut: float = float(os.environ.get("LOSS_CUT", "0.12"))
    # Exit on volume spike ≥ this multiple of the rolling average
    volume_spike_multiplier: float = float(
        os.environ.get("VOLUME_SPIKE_MULTIPLIER", "3.0")
    )

    # ── Operational parameters ─────────────────────────────────────────────
    scan_interval_minutes: int = int(
        os.environ.get("SCAN_INTERVAL_MINUTES", "20")
    )
    max_position_size_usdc: float = float(
        os.environ.get("MAX_POSITION_SIZE_USDC", "100")
    )
    min_liquidity: float = float(os.environ.get("MIN_LIQUIDITY", "1000"))

    # ── API base URLs ──────────────────────────────────────────────────────
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"

    # ── File paths ─────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "trading.db"

    @property
    def wallet_analysis_path(self) -> Path:
        return self.data_dir / "wallet_analysis.json"

    def validate(self) -> None:
        """Raise if required credentials are missing."""
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.poly_api_key:
            missing.append("POLY_API_KEY")
        if not self.poly_private_key:
            missing.append("POLY_PRIVATE_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


# Module-level singleton — import this everywhere
settings = Settings()
