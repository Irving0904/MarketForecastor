"""Alpha Vantage: a second, independent data source used only to
cross-check Yahoo Finance's earnings and news sentiment for the tickers
in a submitted portfolio. Never a primary source, never blocking --
market_data.py's yfinance data is what the rest of the app is built on.

Alpha Vantage's free tier is heavily rate-limited, so results are cached
for 24 hours (far longer than market_data.py's 15-minute ticker cache),
and a missing API key, an exhausted quota, or any request failure
degrades to "no cross-check available for this ticker" rather than an
error -- the profile still builds normally on Yahoo data alone.
"""

import logging

import requests

from market_forecaster.config import get_alpha_vantage_api_key
from market_forecaster.data.cache import TTLCache

logger = logging.getLogger(__name__)

_AV_BASE_URL = "https://www.alphavantage.co/query"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_earnings_cache = TTLCache(_CACHE_TTL_SECONDS)
_sentiment_cache = TTLCache(_CACHE_TTL_SECONDS)

# Beyond this, an EPS difference is treated as a real discrepancy worth
# flagging rather than a rounding/reporting-timing difference between
# two independent data providers.
EPS_DISCREPANCY_TOLERANCE = 0.02
# A discrepancy is only flagged when news sentiment and price action
# point in clearly opposite directions -- not on every small wobble.
BEARISH_SENTIMENT_THRESHOLD = -0.15
BULLISH_SENTIMENT_THRESHOLD = 0.15
STRONG_PRICE_MOVE_PCT = 5.0


def _get(params: dict) -> dict | None:
    api_key = get_alpha_vantage_api_key()
    if not api_key:
        return None
    try:
        response = requests.get(
            _AV_BASE_URL, params={**params, "apikey": api_key}, timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.warning("alpha_vantage: request failed for params=%s", params, exc_info=True)
        return None
    # Alpha Vantage returns HTTP 200 with one of these keys instead of
    # real data when the quota is exhausted or the request is malformed --
    # never a real error status, so this has to be checked explicitly
    # rather than relying on raise_for_status().
    if "Note" in data or "Information" in data or "Error Message" in data:
        logger.warning(
            "alpha_vantage: no usable data (quota/rate-limit or bad request): %s",
            data.get("Note") or data.get("Information") or data.get("Error Message"),
        )
        return None
    return data


def fetch_av_earnings(ticker: str) -> dict | None:
    """Most recent quarterly EPS actual/estimate/surprise from Alpha
    Vantage, shaped to compare directly against
    market_data.py::fetch_earnings's Yahoo equivalent."""
    cached = _earnings_cache.get(ticker)
    if cached is not None:
        return cached
    data = _get({"function": "EARNINGS", "symbol": ticker})
    if not data or not data.get("quarterlyEarnings"):
        return None
    latest = data["quarterlyEarnings"][0]
    try:
        result = {
            "fiscal_date_ending": latest.get("fiscalDateEnding"),
            "reported_eps": float(latest["reportedEPS"]),
            "estimated_eps": float(latest["estimatedEPS"]),
            "surprise_pct": float(latest["surprisePercentage"]),
        }
    except (KeyError, ValueError, TypeError):
        logger.warning("alpha_vantage: unparseable earnings for %s", ticker, exc_info=True)
        return None
    _earnings_cache.set(ticker, result)
    return result


def fetch_av_news_sentiment(ticker: str) -> dict | None:
    """Aggregate news sentiment for this ticker specifically -- averages
    each article's per-ticker sentiment score (ticker_sentiment), not the
    article's overall_sentiment_score, which can reflect other tickers
    mentioned in the same article."""
    cached = _sentiment_cache.get(ticker)
    if cached is not None:
        return cached
    data = _get({"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": "20"})
    if not data or not data.get("feed"):
        return None
    scores = []
    for article in data["feed"]:
        for ts in article.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                try:
                    scores.append(float(ts["ticker_sentiment_score"]))
                except (KeyError, ValueError, TypeError):
                    continue
    if not scores:
        return None
    avg_score = sum(scores) / len(scores)
    result = {"avg_sentiment_score": round(avg_score, 3), "article_count": len(scores)}
    _sentiment_cache.set(ticker, result)
    return result


def cross_check_ticker(ticker: str, yahoo_entry: dict) -> list[dict]:
    """Compares Alpha Vantage's earnings/news data against the same
    ticker's Yahoo-sourced entry (as built by market_data.fetch_yahoo_data).
    Returns a list of discrepancy dicts -- empty if Alpha Vantage has
    nothing to compare (no key, rate-limited, no coverage for this
    ticker) or everything agrees. Never raises."""
    discrepancies = []

    av_earnings = fetch_av_earnings(ticker)
    yahoo_earnings = yahoo_entry.get("earnings") or {}
    if av_earnings and yahoo_earnings.get("last_eps_actual") is not None:
        diff = abs(av_earnings["reported_eps"] - yahoo_earnings["last_eps_actual"])
        if diff > EPS_DISCREPANCY_TOLERANCE:
            discrepancies.append(
                {
                    "ticker": ticker,
                    "type": "earnings_mismatch",
                    "detail": (
                        f"{ticker}: Yahoo reports last EPS of "
                        f"{yahoo_earnings['last_eps_actual']}, Alpha Vantage reports "
                        f"{av_earnings['reported_eps']} for the same period -- worth verifying"
                    ),
                }
            )

    av_sentiment = fetch_av_news_sentiment(ticker)
    price_trend = (yahoo_entry.get("price_history_3mo") or {}).get("pct_change")
    if av_sentiment and price_trend is not None:
        score = av_sentiment["avg_sentiment_score"]
        if score <= BEARISH_SENTIMENT_THRESHOLD and price_trend >= STRONG_PRICE_MOVE_PCT:
            discrepancies.append(
                {
                    "ticker": ticker,
                    "type": "sentiment_divergence",
                    "detail": (
                        f"{ticker}: Alpha Vantage news sentiment is bearish "
                        f"({score:+.2f}) despite the price being up {price_trend:.1f}% "
                        "over 3 months -- news tone and price action disagree"
                    ),
                }
            )
        elif score >= BULLISH_SENTIMENT_THRESHOLD and price_trend <= -STRONG_PRICE_MOVE_PCT:
            discrepancies.append(
                {
                    "ticker": ticker,
                    "type": "sentiment_divergence",
                    "detail": (
                        f"{ticker}: Alpha Vantage news sentiment is bullish "
                        f"({score:+.2f}) despite the price being down {abs(price_trend):.1f}% "
                        "over 3 months -- news tone and price action disagree"
                    ),
                }
            )

    return discrepancies
