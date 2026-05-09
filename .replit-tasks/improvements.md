# Replit Agent Task: polymarket-ai-trader

## Goal
Extend the existing autonomous Polymarket trading bot with Fincept Terminal integration for Bloomberg-level financial data and add a web-based prediction dashboard so Markus can monitor market signals and bot decisions from a browser.

## Tasks
1. **Fincept Terminal integration** (`fincept_client.py`): wrap the Fincept Terminal API (https://fincept.chart.tools — read `FINCEPT_API_KEY` from env); implement:
   - `get_market_news(topic: str, limit=20)` → returns recent news articles with headline, source, sentiment score, published_at
   - `get_macro_indicators(indicators: list[str])` → returns current values for GDP growth, CPI, Fed funds rate, VIX, DXY, etc.
   - `get_sector_sentiment(sector: str)` → returns bullish/bearish sentiment score for a market sector
   - `search_events(query: str)` → searches for prediction-market-relevant events (elections, sports, economic releases)
   - Cache all responses in SQLite for 15 minutes to avoid rate limits
2. **Enhanced market scoring** (update `agents/market_scorer.py` or equivalent): add Fincept data as new scoring signals:
   - News sentiment score (average of top 5 articles for the market topic): weight 15%
   - Macro risk score (VIX level, DXY trend): weight 10%
   - Existing Polymarket signals keep their original weights, scaled down proportionally
   - Document the new scoring formula in a docstring
3. **Prediction dashboard** (`dashboard/`): a web-based FastAPI app (or Flask if FastAPI isn't already in use) that serves:
   - `GET /` → dashboard HTML page (single-page, vanilla JS polling every 30s)
   - `GET /api/markets` → JSON: top 10 scored markets with score breakdown, Fincept news snippets, Claude recommendation
   - `GET /api/positions` → JSON: current open positions with entry price, current price, P&L, exit targets
   - `GET /api/signals` → JSON: last 20 generated trade signals with timestamp, market, direction, confidence
   - `GET /api/news` → JSON: latest Fincept news relevant to open positions
4. **Dashboard UI** (`dashboard/templates/index.html`): dark terminal aesthetic matching existing rich dashboard; sections:
   - **Header**: "Polymarket AI Trader" + live clock + "LIVE" status badge
   - **Positions table**: sortable by P&L; color-coded (green=profit, red=loss); shows Claude recommendation for each
   - **Top Markets table**: score bar chart (CSS width %), news sentiment chip, "Fincept" badge on Fincept-enriched markets
   - **Signal feed**: scrollable list of recent signals with direction arrow, confidence %, timestamp
   - **News ticker**: horizontal scrolling news bar with Fincept headlines at the bottom
   - Auto-refresh every 30 seconds via `setInterval` + fetch
5. **Configuration** (`config/fincept.yaml`): list of macro indicators to monitor, news topics to track (map each to Polymarket category), sector sentiments to pull; load with PyYAML
6. **Claude prompt upgrade**: update the Claude analysis prompt in the existing agent to include Fincept context: prepend top 3 news headlines and current VIX/macro snapshot to the market analysis prompt; this gives Claude richer context for qualitative assessment
7. **Alerting**: add a Telegram alert (reuse existing telegram pattern if present) when: a new high-confidence signal (>80%) is generated, a position hits its price target, or VIX spikes >20% in 24h; read `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from env
8. **Update requirements.txt**: add `fastapi`, `uvicorn`, `pyyaml`, `httpx`, `aiosqlite` (for async SQLite cache); ensure `anthropic>=0.40.0` already present
9. **Startup script** (`run_dashboard.sh`): runs `uvicorn dashboard.app:app --host 0.0.0.0 --port 8080` alongside the existing trading bot main loop; document in README
10. **Update README**: add Fincept setup section (API key, config/fincept.yaml), dashboard access instructions (localhost:8080), new env vars (`FINCEPT_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)

## Tech Stack
- Python 3.10+ (existing)
- Fincept Terminal API (FINCEPT_API_KEY)
- FastAPI + uvicorn (web dashboard)
- SQLite (aiosqlite for async cache)
- Anthropic SDK claude-sonnet-4-6 (existing — already used in bot)
- PyYAML for config
- Existing: py-clob-client, rich, schedule, pandas, numpy

## Deploy Target
Coolify (backend Python service — FastAPI dashboard on port 8080, trading bot as a long-lived process). Never Vercel.

## Done When
- [ ] `fincept_client.py` connects to Fincept API and all 4 methods return data (or graceful mock if API unavailable)
- [ ] Market scoring includes Fincept news sentiment and macro risk as weighted signals
- [ ] FastAPI dashboard starts with `uvicorn dashboard.app:app --port 8080`
- [ ] Dashboard `/api/markets`, `/api/positions`, `/api/signals`, `/api/news` all return JSON
- [ ] `dashboard/templates/index.html` auto-refreshes every 30s and shows all 4 data sections
- [ ] Claude prompt includes Fincept news headlines and VIX snapshot
- [ ] Telegram alerts fire on high-confidence signals (>80%)
- [ ] `config/fincept.yaml` documents monitored indicators and news topics
- [ ] `requirements.txt` updated with all new dependencies
- [ ] README documents Fincept setup, dashboard URL, and all new env vars
