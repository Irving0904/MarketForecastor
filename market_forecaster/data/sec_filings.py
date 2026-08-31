"""SEC EDGAR fetching + chunking for full filing text (10-K, 10-Q, etc).

SEC requires every automated request to identify a real contact via
User-Agent (https://www.sec.gov/os/webmaster-faq#developers) — see
config.get_sec_user_agent(). Pass response.content (bytes) to
BeautifulSoup, not .text — these documents declare encodings SEC's own
server headers get wrong, and letting BeautifulSoup's own detection run
on raw bytes decodes the HTML entities (curly quotes etc.) correctly.
"""

import logging

import requests
from bs4 import BeautifulSoup

from market_forecaster.config import get_sec_user_agent

logger = logging.getLogger(__name__)

_TICKER_TO_CIK: dict[str, str] | None = None


def _headers() -> dict[str, str]:
    return {"User-Agent": get_sec_user_agent()}


def _load_ticker_to_cik() -> dict[str, str]:
    global _TICKER_TO_CIK
    if _TICKER_TO_CIK is not None:
        return _TICKER_TO_CIK
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    mapping = {}
    for entry in resp.json().values():
        mapping[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    _TICKER_TO_CIK = mapping
    logger.info("sec_filings: loaded %d ticker->CIK mappings", len(mapping))
    return mapping


def resolve_cik(ticker: str) -> str | None:
    """ticker -> zero-padded 10-digit CIK string, or None if unknown."""
    return _load_ticker_to_cik().get(ticker.upper())


def fetch_latest_filing(ticker: str, filing_type: str = "10-K") -> dict | None:
    """Most recent filing of filing_type for a ticker. Returns
    {"text", "filing_type", "filing_date", "source_url"} or None if the
    ticker or filing type can't be found."""
    cik = resolve_cik(ticker)
    if cik is None:
        logger.warning("sec_filings: no CIK found for ticker=%s", ticker)
        return None

    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    match_index = next(
        (i for i, form in enumerate(recent["form"]) if form == filing_type), None
    )
    if match_index is None:
        logger.warning(
            "sec_filings: no %s filing found for ticker=%s", filing_type, ticker
        )
        return None

    accession = recent["accessionNumber"][match_index].replace("-", "")
    document = recent["primaryDocument"][match_index]
    filing_date = recent["filingDate"][match_index]
    cik_int = int(cik)
    source_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"
    )

    doc_resp = requests.get(source_url, headers=_headers(), timeout=30)
    doc_resp.raise_for_status()
    soup = BeautifulSoup(doc_resp.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    logger.info(
        "sec_filings: fetched %s %s for %s (%d chars) from %s",
        filing_type,
        filing_date,
        ticker,
        len(text),
        source_url,
    )
    return {
        "text": text,
        "filing_type": filing_type,
        "filing_date": filing_date,
        "source_url": source_url,
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Word-count sliding-window chunking with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks
