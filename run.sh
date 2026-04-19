#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Polymarket AI Trader — launcher
#  Usage:
#    ./run.sh             → live trading (real orders)
#    ./run.sh --dry-run   → simulate only, no real orders
#    ./run.sh --scan      → score markets, no trading
#    ./run.sh --wallets   → analyze poly_data wallet patterns
#    ./run.sh --backtest  → run historical backtest
#    ./run.sh --install   → install/update dependencies only
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
export PIP_USER=false

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Polymarket AI Trader — NYC Tailblazers${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Set up virtualenv if missing ──────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# ── Install / upgrade dependencies ───────────────────────────────────────────
echo -e "${YELLOW}Checking dependencies...${NC}"
PIP_USER=false "$PYTHON" -m pip install --quiet --upgrade pip
PIP_USER=false "$PYTHON" -m pip install --quiet -r requirements.txt
echo -e "${GREEN}✓ Dependencies ready${NC}"

# ── Check .env exists ─────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${RED}ERROR: .env file not found.${NC}"
    echo "Copy .env.example to .env and fill in your API keys."
    exit 1
fi

# ── Parse arguments ───────────────────────────────────────────────────────────
ARG="${1:-}"

case "$ARG" in
    --install)
        echo -e "${GREEN}✓ Install complete.${NC}"
        exit 0
        ;;
    --backtest)
        echo -e "${YELLOW}Running backtest...${NC}"
        "$PYTHON" scripts/backtest.py
        ;;
    --wallets)
        echo -e "${YELLOW}Analyzing wallet patterns from poly_data...${NC}"
        "$PYTHON" scripts/analyze_poly_data.py
        ;;
    --scan)
        echo -e "${YELLOW}Scanning and scoring markets (no trading)...${NC}"
        "$PYTHON" main.py --scan-only
        ;;
    --dry-run)
        echo -e "${YELLOW}Starting in DRY RUN mode (no real orders)...${NC}"
        "$PYTHON" main.py --dry-run
        ;;
    "")
        echo -e "${RED}⚠  LIVE MODE — real orders will be placed on Polymarket${NC}"
        echo -e "${YELLOW}Press Ctrl+C within 5 seconds to cancel...${NC}"
        sleep 5
        "$PYTHON" main.py
        ;;
    *)
        echo "Unknown option: $ARG"
        echo "Usage: ./run.sh [--dry-run | --scan | --wallets | --backtest | --install]"
        exit 1
        ;;
esac
