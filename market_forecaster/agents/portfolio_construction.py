"""Two portfolio-construction agents, used when the advisor asks to build
or recommend a NEW allocation rather than analyze an existing one --
whether or not a client is currently loaded (see orchestrator.py, which
checks `router.construction_router` before the existing paste-a-portfolio /
straight-vs-tot follow-up routing).

Equity Portfolio Agent      -- equities and/or ETFs only.
Diversified Portfolio Agent -- equities, mutual funds, and fixed income.

Both are single pure-reasoning agents (no tools), matching the ToT crew's
agents (tot_crew.py): they reason from the client's stated budget/goals in
the message plus whatever profile summary is already on file, not from a
fresh data-fetch tool call.
"""

import logging

from crewai import Agent, Crew, Task

from market_forecaster.config import get_crew_llm
from market_forecaster.guardrails import DataAccessGuard

logger = logging.getLogger(__name__)


def _profile_block(profile_summary: str) -> str:
    if profile_summary:
        return f"Client profile: {profile_summary}"
    return (
        "No client portfolio has been loaded yet -- treat this as a fresh "
        "account with no existing holdings. Base your recommendation only "
        "on whatever the client request itself states (e.g. a budget or "
        "goal), and general market knowledge."
    )


def build_equity_portfolio_crew(message: str, profile_summary: str) -> Crew:
    DataAccessGuard("Equity Portfolio Agent", [])
    # A full sleeve breakdown (core/growth/satellite tables + rationale)
    # was observed hitting both the 1200- and 1800-token caps and cutting
    # off mid-table -- same failure mode as tot_crew.py's Risk Critic (see
    # that fix's comment): a successful, billed API call that comes back
    # truncated.
    llm = get_crew_llm(max_tokens=2500)
    agent = Agent(
        role="Equity Portfolio Agent",
        goal=(
            "Construct a concrete investment portfolio using only "
            "individual equities and/or ETFs -- no mutual funds, bonds, "
            "or other fixed-income instruments."
        ),
        backstory=(
            "You specialize in pure equity-market portfolios: stock and "
            "ETF selection, sector/position sizing, and growth-vs-value "
            "tilts."
        ),
        llm=llm,
        verbose=False,
    )
    task = Task(
        description=(
            f"{_profile_block(profile_summary)}\n\n"
            f"Client request: {message}\n\n"
            "Build a concrete portfolio allocation using ONLY equities "
            "(individual stocks) and/or ETFs -- do not include mutual "
            "funds, bonds, or any other fixed-income instrument. Name "
            "specific tickers, give an approximate % or $ weight for "
            "each, group them into a small number of sleeves (e.g. "
            "core/growth/satellite), and briefly justify the mix given "
            "the client's stated budget and goal. Be concise: no "
            "restating the request, no filler preamble or repeated "
            "caveats -- keep every piece of substantive reasoning, just "
            "state it efficiently."
        ),
        expected_output=(
            "A concrete equities/ETFs-only portfolio: named tickers, "
            "weights, sleeve groupings, and brief rationale."
        ),
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], verbose=False)


def build_diversified_portfolio_crew(message: str, profile_summary: str) -> Crew:
    DataAccessGuard("Diversified Portfolio Agent", [])
    # Three asset-class breakdowns (equity + mutual fund + fixed income),
    # each with its own table and rationale, need more room than a single
    # equities-only recommendation -- observed hitting the 1200-token cap
    # and cutting off mid-way through the fixed-income section.
    llm = get_crew_llm(max_tokens=2500)
    agent = Agent(
        role="Diversified Portfolio Agent",
        goal=(
            "Recommend a diversified portfolio allocation spanning "
            "equities, mutual funds, and fixed income."
        ),
        backstory=(
            "You specialize in multi-asset-class allocation: balancing "
            "growth (equities), broad low-cost exposure (mutual funds), "
            "and stability/income (fixed income) to match a client's "
            "risk tolerance and time horizon."
        ),
        llm=llm,
        verbose=False,
    )
    task = Task(
        description=(
            f"{_profile_block(profile_summary)}\n\n"
            f"Client request: {message}\n\n"
            "Build a concrete diversified portfolio allocation spanning "
            "all three of: equities, mutual funds, and fixed income. Give "
            "an approximate % weight for each asset class, then name "
            "specific representative holdings (tickers/fund names) "
            "within each, and briefly justify the mix given the "
            "client's stated budget, goal, and (if unstated) a moderate "
            "risk tolerance. Be concise: no restating the request, no "
            "filler preamble or repeated caveats -- keep every piece of "
            "substantive reasoning, just state it efficiently."
        ),
        expected_output=(
            "A concrete diversified portfolio: % weights across equity/"
            "mutual fund/fixed income, representative holdings in each, "
            "and brief rationale."
        ),
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], verbose=False)


def equity_portfolio_pipeline(message: str, session_state: dict) -> str:
    logger.info("equity_portfolio_pipeline: message=%r", message)
    crew = build_equity_portfolio_crew(message, session_state.get("summary", ""))
    answer = str(crew.kickoff())
    logger.info("equity_portfolio_pipeline: answer=%r", answer)
    return answer


def diversified_portfolio_pipeline(message: str, session_state: dict) -> str:
    logger.info("diversified_portfolio_pipeline: message=%r", message)
    crew = build_diversified_portfolio_crew(message, session_state.get("summary", ""))
    answer = str(crew.kickoff())
    logger.info("diversified_portfolio_pipeline: answer=%r", answer)
    return answer
