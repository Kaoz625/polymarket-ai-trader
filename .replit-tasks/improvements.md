# Replit Agent Task Spec

## Instructions for Replit Agent
You are building/improving this project. Read this file carefully before touching any code.
Commit all changes with prefix "replit: " and push to main when done.

## Stack Rules (non-negotiable)
- Static → Cloudflare Pages (never Vercel)
- DB → Supabase self-hosted Docker (never cloud Supabase)
- Auth → NextAuth.js (free, not Auth0/Clerk)
- AI → Claude Sonnet 4.6 via Anthropic API (model: claude-sonnet-4-6)
- Payments (adult) → CCBill or Segpay only

## Improvements To Make
1. **Upgrade Claude model to claude-sonnet-4-6** — Find all places in the codebase where a Claude model is specified (search for "claude-" strings). Update ALL of them to use `claude-sonnet-4-6`. Do not leave any older model strings.
2. **Add Fincept Terminal data integration** — Fincept Terminal provides financial/economic data. Add a new module `agents/fincept_feed.py` that fetches relevant market intelligence. Use the Fincept Terminal API (docs: https://fincept.share.zrok.io/docs or check for a Python SDK). Feed this data as additional context to the prediction agents before they make decisions. If Fincept API is unavailable, fall back gracefully with a log warning.
3. **Improve prediction accuracy** — Add a confidence calibration layer: after each agent makes a prediction, have a second Claude call review the prediction and assign a calibrated confidence score (0-100). Only execute trades where calibrated confidence >= 70. Log raw vs calibrated confidence for analysis.
4. **Add daily P&L reporting** — Create `scripts/daily_report.py` that: queries the SQLite DB for all trades in the last 24 hours, calculates total P&L (realized + unrealized), outputs a formatted rich terminal report with: total trades, win rate, best trade, worst trade, net P&L, and current open positions. Schedule this to run at midnight via the existing `schedule` library. Also save reports to `reports/YYYY-MM-DD.json`.
5. **Add risk controls — max loss per day** — Add a daily loss limit to config. Default: $50/day max loss. In the main trading loop, before every trade: check today's realized P&L from SQLite. If total losses exceed the limit, skip all trades for the rest of the day and log "DAILY LOSS LIMIT REACHED — trading paused". Reset at midnight.
6. **Add position size limits** — Cap any single trade at 5% of total bankroll. Read bankroll from config or calculate from wallet balance. Reject trades that would exceed this cap and log a warning.
7. **Improve logging** — Replace any print() statements with proper Python logging (use `rich` logger or standard `logging` module with file handler). Log to both console and `logs/trading-YYYY-MM-DD.log`. Include timestamps on every log line.

## Do Not Touch
- agents/ directory structure (extend, don't restructure)
- config/ directory (add new keys, don't rename existing ones)
- run.sh and startup.sh scripts

## Definition of Done
- [ ] All improvements implemented and working
- [ ] Claude model updated to claude-sonnet-4-6 everywhere
- [ ] Daily P&L report generates without errors
- [ ] Risk controls prevent trading when daily loss limit hit
- [ ] No Python errors on `python main.py --dry-run` (add dry-run flag if missing)
- [ ] Pushed to main with "replit: " commit prefix
