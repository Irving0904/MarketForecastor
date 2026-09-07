"""Profile Summary Crew: a single Portfolio Analyst agent that turns
already-fetched market data into a client profile summary.

Used to be a two-agent crew: a "Market Data Aggregator" called a tool to
fetch data, then was instructed to echo the tool's raw JSON back verbatim
just so CrewAI's Task.context mechanism could hand it to this summarizer.
That meant ~1000+ tokens/ticker of market data got re-emitted through the
Aggregator's own LLM generation for no reason other than plumbing -- with
a hard scaling ceiling (it silently truncated large portfolios once past
the aggregator's max_tokens cap, dropping tickers with no error surfaced)
and a redundant Yahoo fetch, since orchestrator.py already fetches this
same data directly via fetch_yahoo_data() for cross-checks/alerts/
faithfulness eval before ever building this crew. Passing that raw_data
in as formatted task text instead avoids both: see tot_crew.py's
_format_citable_sources for the same pattern used elsewhere in this repo.
"""

import logging

from crewai import Agent, Crew, Task

from market_forecaster.config import get_crew_llm
from market_forecaster.data.market_data import format_market_data
from market_forecaster.guardrails import DataAccessGuard

logger = logging.getLogger(__name__)


def build_profile_summary_crew(raw_data: dict) -> Crew:
    logger.info("build_profile_summary_crew: tickers=%s", list(raw_data))
    # Validated at construction time, not per-call: raises immediately if
    # the agent is ever granted a tool outside its allow-list.
    DataAccessGuard("Portfolio Analyst", [])
    llm = get_crew_llm()

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

    summarize_task = Task(
        description=(
            "Here is market data already fetched for the client's "
            f"holdings:\n\n{format_market_data(raw_data)}\n\n"
            "Write a concise profile summary covering concentration "
            "risk, sector exposure, dividend income, valuation, and any "
            "red flags. Call out any holding with a notable recent "
            "earnings surprise or a recent news headline that plausibly "
            "explains a large price move. Plain prose, no headers, "
            "under 200 words."
        ),
        expected_output="A short prose paragraph profile summary.",
        agent=summarizer,
    )
    return Crew(
        agents=[summarizer],
        tasks=[summarize_task],
        verbose=False,
    )
