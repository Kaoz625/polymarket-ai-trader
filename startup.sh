#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Polymarket AI Trader — master startup script
#  Launches the 3-process pipeline: brain, executor, exit_monitor
#
#  Usage:
#    bash startup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/.venv/bin/python"
LOGS="$SCRIPT_DIR/logs"
mkdir -p "$LOGS"

# update data from poly_data if available
cd ~/poly_data 2>/dev/null && \
    uv run python -c "from update_utils.process_live import process_live; process_live()" 2>/dev/null || true
cd "$SCRIPT_DIR"

# refresh scan queue
"$PYTHON" scripts/scanner.py

# launch agents as background processes
"$PYTHON" scripts/brain.py &
"$PYTHON" scripts/executor_process.py &
"$PYTHON" scripts/exit_monitor.py &

echo "$(date) — 3 agents live" >> "$LOGS/startup.log"
echo "Startup complete — brain, executor_process, exit_monitor running in background."
echo "PIDs logged. Use 'kill %1 %2 %3' or pkill -f scripts/ to stop."
