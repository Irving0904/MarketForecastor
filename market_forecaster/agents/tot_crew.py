"""Tree-of-Thought Crew: Thought Generator agents -> Critic/Evaluator agent
-> Synthesizer agent.

Agents here are text-only (no tool calls) -- they reason from the profile
summary, the question, and a compact block of already-fetched news/
earnings (see _format_citable_sources), instructed to cite inline when a
claim rests on one of those rather than general reasoning. This is
deliberately lighter-weight than giving them retrieval tools: it reuses
data already fetched for the profile summary instead of adding another
tool-call hop to a pipeline that already has documented backend
flakiness (see README audit trail #5)."""

import logging

from crewai import Agent, Crew, Task

from market_forecaster.config import get_crew_llm
from market_forecaster.guardrails import ConfidenceRouter, DataAccessGuard, critic_output_guardrail

logger = logging.getLogger(__name__)

ANALYST_LENSES = [
    "valuation and downside risk",
    "diversification and concentration risk",
    "growth momentum and upside opportunity",
]


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
    DataAccessGuard("Investment Analyst", [])
    DataAccessGuard("Risk Critic", [])
    DataAccessGuard("Lead Advisor", [])
    llm = get_crew_llm()
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
        llm=llm,
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

    thought_tasks = [
        Task(
            description=(
                f"Client profile: {profile_summary}\n\n"
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

    critic_task = Task(
        description=(
            "Above are three analysts' takes on the same client question, "
            "each from a different lens. Score each 1-10 on rigor and "
            "actionability, and call out the strongest and weakest "
            "reasoning. End your critique with exactly one final line in "
            "the form 'CONFIDENCE: <integer 0-10>' giving your overall "
            "confidence that the synthesized answer will be well-supported "
            "and specific to this client's actual portfolio."
        ),
        expected_output=(
            "A short critique with a score for each analyst, ending with "
            "a 'CONFIDENCE: <0-10>' line."
        ),
        agent=critic,
        context=thought_tasks,
        # Never fails the task (see critic_output_guardrail's docstring) --
        # just logs when the CONFIDENCE line is missing so ConfidenceRouter
        # gets a None score instead of the whole ToT answer being lost.
        guardrail=critic_output_guardrail,
    )

    synthesis_task = Task(
        description=(
            f"Client question: {message}\n\n"
            "Given the analysts' takes and the critic's evaluation above, "
            "weigh them, resolve disagreements, and give one clear final "
            "recommendation with brief reasoning. Keep any (Source: ...) "
            "citations the analysts used for claims you retain in your "
            "answer -- don't drop them, and don't invent new ones."
        ),
        expected_output=(
            "One clear final recommendation with brief reasoning, "
            "preserving any (Source: ...) citations from the analysts' "
            "reasoning that the final answer still relies on."
        ),
        agent=synthesizer,
        context=thought_tasks + [critic_task],
    )

    crew = Crew(
        agents=[generator, critic, synthesizer],
        tasks=thought_tasks + [critic_task, synthesis_task],
        verbose=False,
    )
    return crew, critic_task


def tot_pipeline(message: str, session_state: dict) -> tuple[str, float | None]:
    """Returns (answer, confidence_score) -- confidence_score is the Risk
    Critic's own 0-10 self-assessment (None if it couldn't be parsed even
    after the guardrail's retry), for the caller to run through
    ConfidenceRouter."""
    logger.info("tot_pipeline: message=%r", message)
    crew, critic_task = build_tot_crew(
        message, session_state["summary"], session_state.get("raw_data", {})
    )
    answer = str(crew.kickoff())
    confidence_score = ConfidenceRouter.extract_score(critic_task.output.raw)
    logger.info(
        "tot_pipeline: answer=%r confidence_score=%s", answer, confidence_score
    )
    return answer, confidence_score
