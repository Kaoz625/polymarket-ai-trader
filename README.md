# Polymarket AI Trading System

An autonomous prediction-market trading bot that analyses 86M+ historical
Polymarket trades, scores 500+ live markets every 20 minutes, and executes
a data-driven exit strategy derived from top-wallet behaviour.

---

## What It Does

1. **Wallet intelligence** — downloads the `warproxxx/poly_data` HuggingFace
   dataset (every Polymarket trade ever made), identifies wallets with >60%
   win rate and >50 trades, and extracts their behavioural patterns.

2. **Market scoring** — every 20 minutes it fetches all active markets from
   the Polymarket CLOB + Gamma APIs, scores each one 0–100 across five
   dimensions (liquidity, spread, volume trend, time to resolution,
   sentiment), and surfaces the top candidates.

3. **Claude analysis** — for each top-scored market, `claude-sonnet-4-6`
   provides qualitative assessment: sentiment, fair-value estimate, edge
   description, and a trade recommendation.  Responses are cached 5 minutes.

4. **Signal generation** — combines scores + wallet patterns + Claude output
   to produce directional trade signals with entry price, price target, and
   stop loss.

5. **Execution** — places limit orders via `py-clob-client` and monitors
   positions for the exit conditions below.

6. **Rich dashboard** — terminal UI showing open positions with live P&L,
   top scored markets, balance, and win rate.

---

## Exit Logic

This is the core edge, derived from studying top wallets:

```
EXIT when ANY of these fires first:
  1. Captured ≥ 85% of expected move toward resolution   ← EXIT_THRESHOLD
  2. Position down ≥ 12% from entry                     ← LOSS_CUT
  3. Volume spikes ≥ 3× 24h rolling average             ← VOLUME_SPIKE_MULTIPLIER
```

**Why?** Top wallets exit *before* resolution 91% of the time and capture
a median 86% of the move.  Holding to resolution introduces binary risk
(the market can snap to 0 or 1) while most of the edge is captured early.

Expected move formula:
```
full_move       = 1.0 - entry_price          # for YES side
expected_move   = full_move × 0.86           # median captured_pct
target_price    = entry_price + expected_move
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (orchestrator)                  │
│                         asyncio loop, Rich dashboard            │
└──────┬──────────────┬────────────────┬───────────────┬──────────┘
       │              │                │               │
       ▼              ▼                ▼               ▼
 DataAgent      ScorerAgent     StrategyAgent    ExecutorAgent
 ──────────      ──────────     ─────────────    ─────────────
 HuggingFace    Gamma API       Signal gen       py-clob-client
 poly_data       CLOB API       Exit logic       Order placement
 SQLite cache   Score 0-100    Position mgmt     SQLite trade log
       │
       ▼
 ClaudeAnalyst
 ─────────────
 claude-sonnet-4-6
 Market sentiment
 5-min response cache
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd /path/to/stocks
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 4. Get Polymarket API keys

1. Go to [polymarket.com](https://polymarket.com) and connect a wallet.
2. Navigate to **Settings → API Keys** (or use the CLOB API endpoint
   `POST /auth/api-key` with your wallet signature).
3. You will receive: `api_key`, `api_secret`, `api_passphrase`.
4. Export your wallet's private key — this is `POLY_PRIVATE_KEY`
   (needed to sign on-chain order approvals).

> **WARNING**: Never commit your `.env` file.  It is already in `.gitignore`.

### 5. Get an Anthropic API key

Visit [console.anthropic.com](https://console.anthropic.com) → API Keys →
create a key and paste it as `ANTHROPIC_API_KEY` in `.env`.

---

## Usage

### Live trading

```bash
python main.py
```

### Dry run (simulates trades, no real orders)

```bash
python main.py --dry-run
```

### Scan only (score markets, no trading)

```bash
python main.py --scan-only
```

### Analyse wallet dataset

```bash
python main.py --analyze-wallets
# or run the standalone script:
python scripts/analyze_poly_data.py
python scripts/analyze_poly_data.py --force-refresh --top-n 200
```

### Backtest the exit strategy

```bash
python scripts/backtest.py
python scripts/backtest.py --sample-size 50000 --exit-threshold 0.80
```

---

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Claude analysis |
| `POLY_API_KEY` | — | Polymarket CLOB API key |
| `POLY_API_SECRET` | — | Polymarket CLOB API secret |
| `POLY_API_PASSPHRASE` | — | Polymarket CLOB passphrase |
| `POLY_PRIVATE_KEY` | — | Wallet private key for signing |
| `EXIT_THRESHOLD` | `0.85` | Exit at 85% of expected move |
| `LOSS_CUT` | `0.12` | Cut losers at 12% loss |
| `VOLUME_SPIKE_MULTIPLIER` | `3.0` | Exit on 3× volume spike |
| `SCAN_INTERVAL_MINUTES` | `20` | How often to scan markets |
| `MAX_POSITION_SIZE_USDC` | `100` | Max USDC per position |
| `MIN_LIQUIDITY` | `1000` | Minimum market liquidity |

---

## File Structure

```
stocks/
├── main.py                    # Orchestrator + Rich dashboard
├── requirements.txt
├── .env.example
├── README.md
├── config/
│   └── settings.py            # Typed config dataclass
├── agents/
│   ├── data_agent.py          # HuggingFace poly_data analysis
│   ├── scorer_agent.py        # Market scoring (Gamma + CLOB APIs)
│   ├── strategy_agent.py      # Signal gen + exit logic
│   ├── executor_agent.py      # py-clob-client order management
│   └── claude_analyst.py      # Anthropic API wrapper
├── scripts/
│   ├── analyze_poly_data.py   # Standalone wallet analysis
│   └── backtest.py            # Historical strategy backtester
├── data/                      # SQLite DBs + JSON outputs
└── logs/                      # trading.log
```

---

## Performance Metrics

Target metrics derived from top-wallet analysis:

| Metric | Target | Basis |
|---|---|---|
| Win rate | ≥74% | Observed in top wallets |
| Early-exit rate | ~91% | Exit before resolution |
| Move captured | ~86% | Median of top wallets |
| Loss-cut level | 12% | P75 of losing-trade losses |
| Markets scanned | 500+ every 20 min | Full Gamma API sweep |

> These are targets based on historical data.  Past performance does not
> guarantee future results.  Always start with `--dry-run` and small
> position sizes.

---

## Disclaimer

This software is for educational and research purposes.  Prediction market
trading carries significant financial risk.  Never trade with funds you
cannot afford to lose.  The authors make no guarantees of profitability.
