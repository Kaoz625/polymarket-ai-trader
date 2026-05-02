# polymarket-ai-trader — Project Overview
**Owner:** Kaoz625 | **Stack:** Python, Anthropic API, Polymarket CLOB Client, SQLAlchemy | **Status:** Multi-agent trading system built — needs Fincept data integration, P&L reporting, and risk controls

## What This Is
An AI-powered trading bot for Polymarket prediction markets. Uses 4 specialized AI agents to analyze markets, make predictions, and execute trades with an 85% confidence exit threshold. Incorporates wallet pattern analysis from 86M historical trades for edge detection.

## Current State
Full multi-agent Python system with main orchestrator, agent modules, config, and startup scripts. Uses Anthropic Claude for analysis, py-clob-client for Polymarket API access, SQLite for state. Missing: Fincept Terminal data integration, daily P&L reporting, max loss risk controls per day.

## Tech Stack
- Python 3.10+
- Anthropic SDK (claude-sonnet-4-6)
- py-clob-client (Polymarket API)
- SQLAlchemy + aiosqlite
- pandas, numpy
- aiohttp, websockets
- rich (terminal UI)
- schedule
