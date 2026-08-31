"""Yahoo Finance data fetching, plus a CrewAI tool wrapper around it."""

import json
import logging
import math

import yfinance as yf
from crewai.tools import tool as crew_tool

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


@crew_tool("Fetch Market Data")
def fetch_market_data_tool(tickers: str) -> str:
    """Fetch price, sector, dividend yield, valuation, analyst-rating,
    recent news headlines, trailing 3-month price trend, and earnings
    data for a comma-separated list of stock tickers. Returns JSON."""
    holdings = [{"ticker": t.strip().upper()} for t in tickers.split(",") if t.strip()]
    return json.dumps(fetch_yahoo_data(holdings), default=str)
