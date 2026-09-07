"""Tree-of-Thought Crew: Thought Generator agents -> Critic/Evaluator agent
-> Synthesizer agent.

Agents here are text-only (no tool calls) -- they reason from the profile
summary, the question, and a compact block of already-fetched news/
earnings (see _format_citable_sources), instructed to cite inline when a
claim rests on one of those rather than general reasoning. This is
deliberately lighter-weight than giving them retrieval tools: it reuses
data already fetched for the profile summary instead of adding another
tool-call hop to a pipeline that already has documented backend
flakiness (see README audit trail #5).

When the client's question looks forward-looking (see
looks_like_forecast_question), a fourth "Price Forecast Analyst" joins
the deliberation: it naively extrapolates a price from the trailing
3-month trend, analyst target price, and recent earnings/news, assuming
those factors hold steady -- explicitly a simple extrapolation, not a
statistical or fundamental valuation model. Its output is scored by the
same Risk Critic and folded into the same synthesis as the other three
lenses. It's added conditionally, not unconditionally alongside the
other three, so a question with nothing to do with prediction (e.g. "why
did X move") doesn't pay for a speculative price agent it didn't ask
for."""

import logging
import re

from crewai import Agent, Crew, Task

from market_forecaster.config import get_crew_llm
from market_forecaster.data.market_data import format_market_data
from market_forecaster.guardrails import ConfidenceRouter, DataAccessGuard, critic_output_guardrail

logger = logging.getLogger(__name__)

ANALYST_LENSES = [
    "valuation and downside risk",
    "diversification and concentration risk",
    "growth momentum and upside opportunity",
]

# Cheap heuristic, same style as data/parser.py's looks_like_portfolio --
# deliberately conservative about matching real prediction language
# ("forecast", "price target", "in 6 months") rather than trying to
# understand intent, since a false positive here only costs one extra
# agent call, not a wrong answer.
_FORECAST_PATTERNS = [
    r"\bforecast",
    r"\bpredict",
    r"\bprojection\b",
    r"\bproject(?:ed)?\b",
    r"\bprice target",
    r"\bfuture price",
    r"\bexpected price",
    r"\bwhere (?:will|do you (?:see|think))\b",
    r"\bin \d+\s*(?:day|week|month|year)s?\b",
]
_FORECAST_RE = re.compile("|".join(_FORECAST_PATTERNS), re.IGNORECASE)


def looks_like_forecast_question(message: str) -> bool:
    """Does this ToT question ask for a forward-looking price
    projection, as opposed to general risk/strategy interpretation?"""
    return bool(_FORECAST_RE.search(message))


def _format_citable_sources(raw_data: dict) -> str:
    """Compact, dated, sourced text block the analysts/synthesizer can
    cite from -- built from data already fetched for the profile summary
    (raw_data's news/earnings), not a new retrieval call. Only news and
    earnings are included (not price history/valuation ratios) since
    those are the fields that actually carry a citable date/publisher.
    """
    sections = []
    for ticker, data in raw_data.items():
        if not isinstance(data, dict) or "error" in data:
            continue
        lines = []
        for article in data.get("recent_news") or []:
            title = article.get("title")
            if not title:
                continue
            publisher = article.get("publisher") or "unknown source"
            published = article.get("published") or "undated"
            lines.append(f'- News ({published}, {publisher}): "{title}"')
        earnings = data.get("earnings") or {}
        if earnings.get("last_earnings_date"):
            lines.append(
                f"- Earnings ({earnings['last_earnings_date']}): "
                f"EPS {earnings.get('last_eps_actual')} vs. "
                f"{earnings.get('last_eps_estimate')} estimate "
                f"({earnings.get('last_surprise_pct')}% surprise)"
            )
        if lines:
            sections.append(f"{ticker}:\n" + "\n".join(lines))
    return "\n\n".join(sections)


def build_tot_crew(message: str, profile_summary: str, raw_data: dict) -> tuple[Crew, Task]:
    include_forecast = looks_like_forecast_question(message)
    DataAccessGuard("Investment Analyst", [])
    DataAccessGuard("Risk Critic", [])
    DataAccessGuard("Lead Advisor", [])
    llm = get_crew_llm()
    # The critic scores and critiques all three analysts plus a confidence
    # line -- the default cap was truncating it mid-response (observed
    # hitting exactly 600/600 output tokens and coming back unparseable).
    critic_llm = get_crew_llm(max_tokens=1000)
    # The synthesizer weighs three analyst takes plus a critique into one
    # answer, which needs more room than a single-analyst paragraph.
    synthesis_llm = get_crew_llm(max_tokens=1200)
    citable_sources = _format_citable_sources(raw_data)
    sources_block = (
        f"\n\nRecent news & earnings you may cite (cite inline as "
        f"(Source: <headline or \"Earnings\">, <date>) when you rely on "
        f"one of these rather than general reasoning):\n{citable_sources}"
        if citable_sources
        else ""
    )

    generator = Agent(
        role="Investment Analyst",
        goal=(
            "Generate one independent, well-reasoned line of thinking on "
            "a client question from a specific lens."
        ),
        backstory=(
            "You are one of several analysts on a team, each assigned a "
            "distinct angle so the team covers the full picture."
        ),
        llm=llm,
        verbose=False,
    )
    critic = Agent(
        role="Risk Critic",
        goal=(
            "Score and critique each analyst's reasoning for rigor, "
            "actionability, and relevance to the client's actual "
            "portfolio."
        ),
        backstory=(
            "You are a skeptical reviewer whose job is to catch weak "
            "reasoning before it reaches the client."
        ),
        llm=critic_llm,
        verbose=False,
    )
    synthesizer = Agent(
        role="Lead Advisor",
        goal=(
            "Combine the analysts' reasoning and the critic's evaluation "
            "into one clear final recommendation."
        ),
        backstory=(
            "You have the final say and must resolve disagreements into "
            "a single actionable recommendation."
        ),
        llm=synthesis_llm,
        verbose=False,
    )
    forecast_analyst = None
    if include_forecast:
        DataAccessGuard("Price Forecast Analyst", [])
        forecast_analyst = Agent(
            role="Price Forecast Analyst",
            goal=(
                "Project a forward-looking price estimate for the "
                "client's relevant holdings by naively extrapolating "
                "their trailing price trend, analyst target price, and "
                "recent earnings/news, explicitly assuming those "
                "factors hold steady."
            ),
            backstory=(
                "You produce simple, assumption-driven price "
                "projections, not statistical or fundamental valuation "
                "models -- you are always explicit about the "
                "assumptions a projection rests on and that real prices "
                "can diverge from it at any time."
            ),
            llm=llm,
            verbose=False,
        )

    profile_block = (
        f"Client profile: {profile_summary}"
        if profile_summary
        else (
            "No client portfolio has been loaded yet -- treat this as a "
            "fresh account with no existing holdings. Base your reasoning "
            "only on whatever the client question itself states (e.g. a "
            "budget or goal), and general market knowledge."
        )
    )

    thought_tasks = [
        Task(
            description=(
                f"{profile_block}\n\n"
                f"Client question: {message}"
                f"{sources_block}\n\n"
                f"Give one independent line of reasoning from the lens of "
                f"{lens}, and a tentative recommendation, in under 100 "
                "words."
            ),
            expected_output=(
                "A short paragraph of reasoning and a tentative "
                "recommendation."
            ),
            agent=generator,
        )
        for lens in ANALYST_LENSES
    ]

    forecast_task = None
    if include_forecast:
        forecast_task = Task(
            description=(
                f"Client question: {message}\n\n"
                "Market data for the client's holdings (price, "
                "valuation, analyst rating/target, trailing 3-month "
                "trend, and recent earnings/news):\n\n"
                f"{format_market_data(raw_data)}\n\n"
                "For the holding(s) the question is about, project a "
                "price forecast over the horizon implied by the "
                "question (default to 3-6 months if none is stated). "
                "Base the projection explicitly on: the trailing "
                "3-month price trend continuing at the same rate, the "
                "analyst target price, and whether the latest earnings "
                "surprise and recent news suggest that trend is "
                "accelerating, reversing, or holding steady. State your "
                "assumptions plainly, give a specific projected price "
                "or range, and note this is a naive extrapolation, not "
                "a statistical or fundamental model, and that real "
                "prices can diverge from it at any time."
            ),
            expected_output=(
                "A stated horizon, a specific projected price or range "
                "per relevant holding, the trend/rating/earnings basis "
                "for it, and an explicit naive-extrapolation caveat."
            ),
            agent=forecast_analyst,
        )

    context_tasks = thought_tasks + ([forecast_task] if include_forecast else [])
    analyst_intro = (
        "Above are three analysts' takes on the same client question, "
        "each from a different lens, plus a Price Forecast Analyst's "
        "price projection."
        if include_forecast
        else "Above are three analysts' takes on the same client "
        "question, each from a different lens."
    )
    forecast_scoring_note = (
        " For the Price Forecast Analyst, judge whether its stated "
        "assumptions are reasonable and clearly tied to the trend/"
        "rating/earnings data, not whether the price will actually be "
        "hit."
        if include_forecast
        else ""
    )
    critic_task = Task(
        description=(
            f"{analyst_intro} Score each 1-10 on rigor and "
            f"actionability, and call out the strongest and weakest "
            f"reasoning.{forecast_scoring_note} End your critique with "
            "exactly one final line in the form 'CONFIDENCE: <integer "
            "0-10>' giving your overall confidence that the synthesized "
            "answer will be well-supported and specific to this "
            "client's actual portfolio."
        ),
        expected_output=(
            "A short critique with a score for each analyst, ending with "
            "a 'CONFIDENCE: <0-10>' line."
        ),
        agent=critic,
        context=context_tasks,
        # Never fails the task (see critic_output_guardrail's docstring) --
        # just logs when the CONFIDENCE line is missing so ConfidenceRouter
        # gets a None score instead of the whole ToT answer being lost.
        guardrail=critic_output_guardrail,
    )

    forecast_synthesis_note = (
        " If a price forecast was produced, include its projected "
        "price/range and the assumptions it rests on in your final "
        "answer, clearly labeled as a projection, not a guarantee."
        if include_forecast
        else ""
    )
    synthesis_task = Task(
        description=(
            f"Client question: {message}\n\n"
            "Given the analysts' takes and the critic's evaluation above, "
            "weigh them, resolve disagreements, and give one clear final "
            "recommendation with brief reasoning. Keep any (Source: ...) "
            "citations the analysts used for claims you retain in your "
            "answer -- don't drop them, and don't invent new ones. Be "
            "concise: no restating the question, no filler preamble or "
            "repeated caveats -- keep every piece of substantive "
            f"reasoning, just state it efficiently.{forecast_synthesis_note}"
        ),
        expected_output=(
            "One clear final recommendation with brief reasoning, "
            "preserving any (Source: ...) citations from the analysts' "
            "reasoning that the final answer still relies on."
        ),
        agent=synthesizer,
        context=context_tasks + [critic_task],
    )

    agents = [generator, critic, synthesizer]
    tasks = thought_tasks
    if include_forecast:
        agents.append(forecast_analyst)
        tasks = tasks + [forecast_task]
    tasks = tasks + [critic_task, synthesis_task]

    crew = Crew(agents=agents, tasks=tasks, verbose=False)
    return crew, critic_task


def tot_pipeline(message: str, session_state: dict) -> tuple[str, float | None]:
    """Returns (answer, confidence_score) -- confidence_score is the Risk
    Critic's own 0-10 self-assessment (None if it couldn't be parsed even
    after the guardrail's retry), for the caller to run through
    ConfidenceRouter."""
    logger.info("tot_pipeline: message=%r", message)
    crew, critic_task = build_tot_crew(
        message, session_state.get("summary", ""), session_state.get("raw_data", {})
    )
    answer = str(crew.kickoff())
    confidence_score = ConfidenceRouter.extract_score(critic_task.output.raw)
    logger.info(
        "tot_pipeline: answer=%r confidence_score=%s", answer, confidence_score
    )
    return answer, confidence_score
