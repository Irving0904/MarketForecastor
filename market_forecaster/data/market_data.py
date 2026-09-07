"""Yahoo Finance data fetching."""

import logging
import math

import yfinance as yf

from market_forecaster.data.cache import TTLCache

logger = logging.getLogger(__name__)

TICKER_CACHE_TTL_SECONDS = 15 * 60
_ticker_cache = TTLCache(TICKER_CACHE_TTL_SECONDS)


def _safe_float(value) -> float | None:
    """yfinance surfaces missing numbers as NaN, which isn't valid JSON."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 2)


def fetch_news(t: yf.Ticker, limit: int = 5) -> list[dict]:
    """Recent news headlines for a ticker."""
    try:
        items = t.news or []
    except Exception:
        logger.warning("fetch_news: failed for %s", t.ticker, exc_info=True)
        return []
    articles = []
    for item in items[:limit]:
        content = item.get("content", {}) or {}
        provider = content.get("provider", {}) or {}
        canonical = content.get("canonicalUrl", {}) or {}
        articles.append(
            {
                "title": content.get("title"),
                "publisher": provider.get("displayName"),
                "published": content.get("pubDate"),
                "url": canonical.get("url"),
            }
        )
    return articles


def fetch_price_history(t: yf.Ticker, period: str = "3mo") -> dict:
    """Summary stats over a trailing window — not the raw OHLCV series,
    to keep this compact enough to hand to an LLM."""
    try:
        hist = t.history(period=period)
    except Exception:
        logger.warning("fetch_price_history: failed for %s", t.ticker, exc_info=True)
        return {}
    if hist.empty:
        return {}
    start_price = float(hist["Close"].iloc[0])
    end_price = float(hist["Close"].iloc[-1])
    return {
        "period": period,
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "pct_change": (
            round((end_price / start_price - 1) * 100, 1) if start_price else None
        ),
        "period_high": round(float(hist["High"].max()), 2),
        "period_low": round(float(hist["Low"].min()), 2),
        "avg_volume": int(hist["Volume"].mean()),
    }


def fetch_earnings(t: yf.Ticker) -> dict:
    """Most recent reported earnings (with surprise %) and the next
    upcoming earnings date, if known."""
    try:
        ed = t.earnings_dates
    except Exception:
        logger.warning("fetch_earnings: failed for %s", t.ticker, exc_info=True)
        return {}
    if ed is None or ed.empty:
        return {}

    result = {}
    reported = ed[ed["Reported EPS"].notna()].sort_index(ascending=False)
    if not reported.empty:
        row = reported.iloc[0]
        result["last_earnings_date"] = str(reported.index[0].date())
        result["last_eps_estimate"] = _safe_float(row.get("EPS Estimate"))
        result["last_eps_actual"] = _safe_float(row.get("Reported EPS"))
        result["last_surprise_pct"] = _safe_float(row.get("Surprise(%)"))

    upcoming = ed[ed["Reported EPS"].isna()].sort_index(ascending=True)
    if not upcoming.empty:
        row = upcoming.iloc[0]
        result["next_earnings_date"] = str(upcoming.index[0].date())
        result["next_eps_estimate"] = _safe_float(row.get("EPS Estimate"))

    return result


def fetch_yahoo_data(holdings: list[dict]) -> dict:
    """Pull price, dividend, valuation, analyst-rating, recent news,
    trailing price trend, and earnings data per ticker.

    Cached per ticker for TICKER_CACHE_TTL_SECONDS — a duplicate ticker
    (within this portfolio, or held by another client, or re-fetched
    within the cache window) is served from cache instead of hitting
    Yahoo Finance again."""
    data = {}
    for h in holdings:
        ticker = h["ticker"]
        if ticker in data:
            continue  # duplicate within this same portfolio

        cached = _ticker_cache.get(ticker)
        if cached is not None:
            logger.info("fetch_yahoo_data: cache hit for %s", ticker)
            data[ticker] = cached
            continue

        try:
            t = yf.Ticker(ticker)
            fast = t.fast_info
            info = t.info
            # fast_info's fields are lazy — accessing t.fast_info above
            # doesn't itself trigger a fetch or raise for an invalid
            # ticker; only .get() on individual fields does, so the whole
            # entry has to be built inside this try, not after it.
            entry = {
                "price": fast.get("lastPrice"),
                "market_cap": fast.get("marketCap"),
                "year_change_pct": round((fast.get("yearChange") or 0) * 100, 1),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "dividend_yield_pct": round(info.get("dividendYield") or 0, 2),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "analyst_recommendation": info.get("recommendationKey"),
                "target_mean_price": info.get("targetMeanPrice"),
                "recent_news": fetch_news(t),
                "price_history_3mo": fetch_price_history(t),
                "earnings": fetch_earnings(t),
            }
        except Exception as exc:
            logger.warning("fetch_yahoo_data: failed for %s", ticker, exc_info=True)
            data[ticker] = {"error": str(exc)}
            continue
        data[ticker] = entry
        _ticker_cache.set(ticker, entry)
        logger.info("fetch_yahoo_data: fetched fresh data for %s", ticker)
    return data


_MAX_NEWS_PER_TICKER = 3


def format_market_data(raw_data: dict) -> str:
    """Renders fetch_yahoo_data's output as compact per-ticker text, for
    handing to an LLM as plain task context (never re-echoed through the
    LLM's own generation -- see profile_crew.py and tot_crew.py, both of
    which use this as their agents' factual basis: price/valuation/
    dividend/analyst-rating snapshot, trailing 3-month trend, and recent
    earnings/news)."""
    sections = []
    for ticker, entry in raw_data.items():
        if not isinstance(entry, dict):
            continue
        if "error" in entry:
            sections.append(f"{ticker}: data unavailable ({entry['error']})")
            continue
        lines = [
            f"price=${entry.get('price')}, market_cap={entry.get('market_cap')}, "
            f"year_change={entry.get('year_change_pct')}%",
            f"sector={entry.get('sector')}, industry={entry.get('industry')}",
            f"dividend_yield={entry.get('dividend_yield_pct')}%, "
            f"trailing_pe={entry.get('trailing_pe')}, "
            f"forward_pe={entry.get('forward_pe')}",
            f"analyst_recommendation={entry.get('analyst_recommendation')}, "
            f"target_mean_price={entry.get('target_mean_price')}",
        ]
        history = entry.get("price_history_3mo") or {}
        if history:
            lines.append(
                f"3mo_price_change={history.get('pct_change')}% "
                f"(high={history.get('period_high')}, low={history.get('period_low')})"
            )
        earnings = entry.get("earnings") or {}
        if earnings.get("last_earnings_date"):
            lines.append(
                f"last_earnings ({earnings['last_earnings_date']}): "
                f"EPS {earnings.get('last_eps_actual')} vs. "
                f"{earnings.get('last_eps_estimate')} estimate "
                f"({earnings.get('last_surprise_pct')}% surprise)"
            )
        if earnings.get("next_earnings_date"):
            lines.append(f"next_earnings_date={earnings['next_earnings_date']}")
        for article in (entry.get("recent_news") or [])[:_MAX_NEWS_PER_TICKER]:
            title = article.get("title")
            if not title:
                continue
            lines.append(
                f'news ({article.get("published") or "undated"}, '
                f'{article.get("publisher") or "unknown source"}): "{title}"'
            )
        sections.append(f"{ticker}:\n  " + "\n  ".join(lines))
    return "\n\n".join(sections)
