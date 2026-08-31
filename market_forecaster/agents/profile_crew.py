"""Profile Summary Crew: a Data Aggregator agent (fetches Yahoo data via a
tool) hands off to a Profile Summarizer agent."""

import logging

from crewai import Agent, Crew, Task

from market_forecaster.config import get_crew_llm
from market_forecaster.data.market_data import fetch_market_data_tool
from market_forecaster.guardrails import DataAccessGuard

logger = logging.getLogger(__name__)


def build_profile_summary_crew(tickers: list[str]) -> Crew:
    logger.info("build_profile_summary_crew: tickers=%s", tickers)
    # Validated at construction time, not per-call: raises immediately if
    # either agent is ever granted a tool outside its allow-list.
    DataAccessGuard("Market Data Aggregator", ["Fetch Market Data"])
    DataAccessGuard("Portfolio Analyst", [])
    llm = get_crew_llm()
    # The aggregator echoes back the tool's raw JSON verbatim, which now
    # includes news/price-history/earnings per ticker — easily well past
    # the default 600-token cap for more than one or two holdings, which
    # silently truncated later tickers out of the summary.
    aggregator_llm = get_crew_llm(max_tokens=6000)

    aggregator = Agent(
        role="Market Data Aggregator",
        goal="Fetch accurate, current market data for the client's holdings.",
        backstory=(
            "You are meticulous about pulling raw, unbiased market data "
            "before any analysis happens."
        ),
        tools=[fetch_market_data_tool],
        llm=aggregator_llm,
        verbose=False,
    )
    summarizer = Agent(
        role="Portfolio Analyst",
        goal="Turn raw market data into a concise client profile summary.",
        backstory=(
            "You write plain-English portfolio assessments covering "
            "concentration, sector exposure, income, and valuation risk."
        ),
        llm=llm,
        verbose=False,
    )

    aggregate_task = Task(
        description=(
            "Use the Fetch Market Data tool to pull data for these "
            f"tickers: {', '.join(tickers)}. Return the raw JSON data "
            "exactly as the tool gives it to you."
        ),
        expected_output="A JSON object of market data keyed by ticker.",
        agent=aggregator,
    )
    summarize_task = Task(
        description=(
            "Given the market data above, write a concise profile summary "
            "covering concentration risk, sector exposure, dividend "
            "income, valuation, and any red flags. Call out any holding "
            "with a notable recent earnings surprise or a recent news "
            "headline that plausibly explains a large price move. Plain "
            "prose, no headers, under 200 words."
        ),
        expected_output="A short prose paragraph profile summary.",
        agent=summarizer,
        context=[aggregate_task],
    )
    return Crew(
        agents=[aggregator, summarizer],
        tasks=[aggregate_task, summarize_task],
        verbose=False,
    )
