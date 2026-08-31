"""Per-ticker change detection and portfolio-level rollup.

Compares a client's newly-fetched market data against their own
previously-persisted snapshot (profile_state["raw_data"] from before this
turn overwrites it). No alert is fabricated: a ticker with no prior
snapshot (first-time submission) or held on only one side of the diff
(newly added / dropped since last time) has nothing to compare against
and is silently excluded, never presented as "unchanged."
"""

import logging

logger = logging.getLogger(__name__)

PRICE_MOVE_MEDIUM_PCT = 5.0
PRICE_MOVE_HIGH_PCT = 10.0


def _classify_ticker_change(ticker: str, old: dict, new: dict) -> dict | None:
    old_rating = old.get("analyst_recommendation")
    new_rating = new.get("analyst_recommendation")
    if old_rating and new_rating and old_rating != new_rating:
        return {
            "ticker": ticker,
            "type": "rating_change",
            "severity": "high",
            "detail": f"{ticker}: analyst rating changed from {old_rating} to {new_rating}",
        }

    old_price = old.get("price")
    new_price = new.get("price")
    if isinstance(old_price, (int, float)) and isinstance(new_price, (int, float)) and old_price:
        pct = (new_price - old_price) / old_price * 100
        if abs(pct) >= PRICE_MOVE_HIGH_PCT:
            severity = "high"
        elif abs(pct) >= PRICE_MOVE_MEDIUM_PCT:
            severity = "medium"
        else:
            return None
        direction = "up" if pct > 0 else "down"
        return {
            "ticker": ticker,
            "type": "price_move",
            "severity": severity,
            "detail": f"{ticker}: price moved {direction} {abs(pct):.1f}% since your last check-in",
        }
    return None


def build_portfolio_alerts(old_raw_data: dict, new_raw_data: dict) -> list[dict]:
    """Only diffs tickers present in both snapshots -- a ticker on just
    one side (newly added, or dropped since last time) is excluded, not
    treated as a change."""
    alerts = []
    shared_tickers = set(old_raw_data) & set(new_raw_data)
    for ticker in sorted(shared_tickers):
        old = old_raw_data.get(ticker) or {}
        new = new_raw_data.get(ticker) or {}
        if "error" in old or "error" in new:
            continue
        alert = _classify_ticker_change(ticker, old, new)
        if alert:
            alerts.append(alert)
    return alerts


def format_rollup_line(alerts: list[dict], total_holdings: int) -> str:
    """Empty string if there's nothing to report -- callers just prepend
    this directly, so no alerts means no visible change to the reply."""
    if not alerts:
        return ""
    high_count = sum(1 for a in alerts if a["severity"] == "high")
    high_note = f", {high_count} high-impact" if high_count else ""
    bullets = "\n".join(f"- {a['detail']}" for a in alerts)
    return (
        f"📋 **{len(alerts)} of your {total_holdings} holdings had changes "
        f"since your last check-in{high_note}:**\n{bullets}\n\n"
    )
