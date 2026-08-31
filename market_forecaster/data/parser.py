"""Parsing a pasted portfolio (CSV or ticker list) into holdings.

`HOLDING_PAIR_RE` (ticker+shares, e.g. "AAPL,100") is case-insensitive and
tolerant of stray whitespace (e.g. "orcl ,800" or "NVDA,540 MSFT,480" on
one line) — both are real inputs that broke the original strict CSV-line
/ word-count heuristics. `TICKER_RE` stays uppercase-only: it's the
fallback for a bare ticker list with no share counts, and case is the
only signal keeping it from matching ordinary short English words in a
follow-up question (e.g. "is", "my", "risk").
"""

import re

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
HOLDING_PAIR_RE = re.compile(r"\b([A-Za-z]{1,5})\s*,\s*(\d+(?:\.\d+)?)\b")


def looks_like_portfolio(message: str) -> bool:
    """Cheap first-layer router: portfolio submission vs. follow-up question."""
    if HOLDING_PAIR_RE.search(message):
        return True
    tickers = TICKER_RE.findall(message)
    words = message.split()
    return len(tickers) >= 3 and len(tickers) / max(len(words), 1) > 0.3


def parse_portfolio(message: str) -> list[dict]:
    """Parse pasted CSV text, ticker+shares pairs, or a loose ticker list
    into holdings — any of which may be on one line or many."""
    pairs = HOLDING_PAIR_RE.findall(message)
    if pairs:
        return [
            {"ticker": ticker.upper(), "raw": [ticker, shares]}
            for ticker, shares in pairs
        ]
    return [
        {"ticker": ticker, "raw": [ticker]} for ticker in TICKER_RE.findall(message)
    ]
