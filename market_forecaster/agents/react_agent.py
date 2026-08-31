"""LangChain tool-calling ReAct agent for single-fact holding lookups,
SEC filing search, and semantic search over a client's past conversations.
"""

import json
import logging

from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool

from market_forecaster.agents.utils import extract_text
from market_forecaster.config import get_chat_model
from market_forecaster.data import vector_store
from market_forecaster.data.sec_filings import chunk_text, fetch_latest_filing
from market_forecaster.guardrails import DataAccessGuard

logger = logging.getLogger(__name__)


@lc_tool
def search_filings(ticker: str, query: str) -> str:
    """Search a ticker's most recent 10-K filing for passages relevant to
    query — use for questions that need actual filing content (risk
    factors, business description, etc), not just summary market data.
    Fetches and embeds the filing on first use for that ticker; later
    calls (any client, any session) reuse the cached chunks."""
    ticker = ticker.upper()
    logger.debug("search_filings tool called ticker=%r query=%r", ticker, query)

    if not vector_store.has_chunks("sec_filings", where={"ticker": ticker}):
        filing = fetch_latest_filing(ticker, "10-K")
        if filing is None:
            return json.dumps({"error": f"No 10-K filing found for {ticker}"})
        chunks = chunk_text(filing["text"])
        ids = [
            f"{ticker}_{filing['filing_type']}_{filing['filing_date']}_{i}"
            for i in range(len(chunks))
        ]
        metadatas = [
            {
                "ticker": ticker,
                "filing_type": filing["filing_type"],
                "filing_date": filing["filing_date"],
                "chunk_index": i,
                "source_url": filing["source_url"],
            }
            for i in range(len(chunks))
        ]
        vector_store.add_chunks("sec_filings", ids, chunks, metadatas)

    matches = vector_store.query(
        "sec_filings", query, n_results=3, where={"ticker": ticker}
    )
    return json.dumps(matches, default=str)


def build_react_agent(session_state: dict, client_id: str | None):
    """Builds the ReAct agent without invoking it -- factored out of
    react_pipeline so evaluators.py can invoke the same agent and inspect
    which tool it actually called, instead of only the final text answer
    react_pipeline returns."""
    raw_data = session_state["raw_data"]
    tickers = ", ".join(session_state["tickers"])

    @lc_tool
    def get_holding_data(ticker: str) -> str:
        """Get current market data for one ticker in the client's
        portfolio: price, sector, dividend yield, PE ratios, analyst
        rating, target price, recent news headlines, trailing 3-month
        price trend, and earnings (last reported + next upcoming)."""
        logger.debug("get_holding_data tool called with ticker=%r", ticker)
        result = raw_data.get(ticker.upper(), {"error": f"No data for {ticker}"})
        return json.dumps(result, default=str)

    @lc_tool
    def search_client_history(query: str) -> str:
        """Search this client's past conversation history for passages
        relevant to query — use for questions like 'what did we discuss
        about X before' or 'what did we tell them about Y'."""
        logger.debug("search_client_history tool called query=%r", query)
        if not client_id:
            return json.dumps({"error": "No active client to search history for"})
        matches = vector_store.query(
            "client_history", query, n_results=5, where={"client_id": client_id}
        )
        return json.dumps(matches, default=str)

    DataAccessGuard(
        "ReAct portfolio assistant",
        ["get_holding_data", "search_filings", "search_client_history"],
    )
    llm = get_chat_model()
    return create_agent(
        llm,
        tools=[get_holding_data, search_filings, search_client_history],
        system_prompt=(
            "You are a ReAct-style portfolio assistant. The client's "
            f"holdings are: {tickers}. Use get_holding_data to look up "
            "any figures you need before answering, including recent "
            "news, price trend, and earnings data when explaining why a "
            "holding moved. Use search_filings when a question needs "
            "actual 10-K content (risk factors, business description) "
            "rather than summary data. Use search_client_history for "
            "questions about what was discussed with this client before. "
            "Answer directly and concisely, and cite specific news "
            "headlines, filing passages, or earnings figures when they "
            "explain the answer rather than speculating."
        ),
    )


def react_pipeline(message: str, session_state: dict, client_id: str | None) -> str:
    logger.info(
        "react_pipeline: message=%r tickers=%s client_id=%s",
        message,
        session_state.get("tickers"),
        client_id,
    )
    agent = build_react_agent(session_state, client_id)
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    answer = extract_text(result["messages"][-1].content)
    logger.info("react_pipeline: answer=%r", answer)
    return answer
