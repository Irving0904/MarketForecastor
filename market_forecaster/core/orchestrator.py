"""Top-level session orchestration: decides which pipeline handles each turn.

- A cheap heuristic router (`looks_like_portfolio`) decides portfolio
  submission vs. follow-up question before any LLM/agent call happens.
- Portfolio submission runs the CrewAI Profile Summary Crew.
- Follow-up questions go through the LangChain router chain, which picks
  "straight" (LangChain ReAct agent) or "tot" (CrewAI ToT Crew).

Every stage logs through the standard `logging` module (see main.py for
the handler config) so a failure's log line names the exact function that
raised, instead of only a generic "Error" surfacing in the UI.
"""

import json
import logging

from market_forecaster.agents.profile_crew import build_profile_summary_crew
from market_forecaster.agents.react_agent import react_pipeline
from market_forecaster.agents.router import router_agent
from market_forecaster.agents.tot_crew import tot_pipeline
from market_forecaster.core.alerts import build_portfolio_alerts, format_rollup_line
from market_forecaster.data.market_data import fetch_yahoo_data
from market_forecaster.data.parser import looks_like_portfolio, parse_portfolio
from market_forecaster.evaluators import FaithfulnessEvaluator, RelevancyEvaluator
from market_forecaster.guardrails import ConfidenceRouter, ContentPolicyGuard

logger = logging.getLogger(__name__)


def _report(progress, fraction: float, desc: str) -> None:
    """progress is any callable(fraction, desc=...) — e.g. gr.Progress() —
    or None. Kept duck-typed so this module never needs to import gradio."""
    if progress is not None:
        progress(fraction, desc=desc)


def _check_faithfulness(summary: str, raw_data: dict) -> str:
    """Evaluator + guardrail applied to the portfolio summary: measures
    whether its claims are grounded in the raw market data it was built
    from, and annotates rather than blocks on a failure — the summary is
    still the best answer available, it just needs a caveat."""
    try:
        result = FaithfulnessEvaluator.check(summary, json.dumps(raw_data, default=str))
    except Exception:
        logger.exception("respond: faithfulness check itself failed, skipping")
        return summary
    if not result.passed:
        logger.warning(
            "respond: profile summary failed faithfulness check pass_rate=%.2f",
            result.pass_rate,
        )
        return (
            "[Some claims in this summary weren't clearly supported by the "
            "fetched market data — verify before relying on it.]\n\n" + summary
        )
    return summary


def _check_answer(message: str, answer: str) -> str:
    """Content-policy and relevancy checks applied to every ReAct/ToT
    answer before it reaches the advisor. Content policy can annotate;
    relevancy is logged as a measurement signal (its failure rate isn't
    itself actionable per-answer the way advice-language is)."""
    policy_result = ContentPolicyGuard.check(answer)
    if policy_result.flagged:
        logger.warning(
            "respond: content policy flagged phrases=%s", policy_result.matched_phrases
        )
    answer = ContentPolicyGuard.annotate(answer, policy_result)

    try:
        relevancy = RelevancyEvaluator.check(message, answer)
        if not relevancy.passed:
            logger.warning(
                "respond: relevancy check failed pass_rate=%.2f for message=%r",
                relevancy.pass_rate,
                message,
            )
    except Exception:
        logger.exception("respond: relevancy check itself failed, skipping")

    return answer


def respond(
    message: str,
    history: list,
    profile_state: dict | None,
    client_id: str | None = None,
    progress=None,
):
    """Returns (reply, updated_profile_state, route_tag) — route_tag labels
    which pipeline handled the turn, for the Trace panel. client_id scopes
    the ReAct agent's search_client_history tool to this client only.
    progress, if given, is called with (fraction, desc=...) at each stage
    boundary to drive a live status indicator."""
    profile_state = profile_state or {}

    if looks_like_portfolio(message):
        holdings = parse_portfolio(message)
        if not holdings:
            logger.warning("respond: no tickers found in message=%r", message)
            return (
                "I couldn't find any tickers in that — try pasting a CSV "
                "(ticker,shares,...) or a plain list of symbols.",
                profile_state,
                "error",
            )
        tickers = [h["ticker"] for h in holdings]
        # Captured before this turn's fetch overwrites profile_state --
        # the only source for "what changed since last time" is the
        # client's own previously-persisted snapshot, passed in as
        # profile_state by the caller.
        old_raw_data = profile_state.get("raw_data")
        try:
            _report(progress, 0.1, f"Step 1/3: Fetching market data for {', '.join(tickers)}...")
            raw_data = fetch_yahoo_data(holdings)
            _report(progress, 0.4, "Step 2/3: Running Profile Summary Crew (Data Aggregator -> Portfolio Analyst)...")
            crew = build_profile_summary_crew(tickers)
            summary = str(crew.kickoff())
            summary = _check_faithfulness(summary, raw_data)
            _report(progress, 0.9, "Step 3/3: Finalizing profile...")
        except RuntimeError as exc:
            logger.warning("respond: missing API key building profile: %s", exc)
            return str(exc), profile_state, "error"
        except Exception:
            logger.exception(
                "respond: profile summary crew failed for tickers=%s", tickers
            )
            return (
                "The analysis crew hit an unexpected error building your "
                "profile — this can happen intermittently with the LLM "
                "backend. Try pasting your portfolio again.",
                profile_state,
                "error",
            )
        profile_state = {
            "holdings": holdings,
            "tickers": tickers,
            "raw_data": raw_data,
            "summary": summary,
        }
        rollup = ""
        if old_raw_data:
            alerts = build_portfolio_alerts(old_raw_data, raw_data)
            rollup = format_rollup_line(alerts, len(tickers))
        logger.info("respond: profile built for tickers=%s", tickers)
        return (
            f"{rollup}Got it — here's your profile:\n\n{summary}\n\n"
            "Ask me anything about it.",
            profile_state,
            "portfolio",
        )

    if not profile_state.get("summary"):
        return (
            "Paste your portfolio first (CSV or a ticker list) so I have "
            "something to analyze.",
            profile_state,
            "prompt",
        )

    route = None
    try:
        _report(progress, 0.1, "Step 1/4: Checking your question...")
        _report(progress, 0.3, "Step 2/4: Classifying question type...")
        route = router_agent(message, profile_state["summary"])
        if route == "straight":
            _report(progress, 0.5, "Step 3/4: Running ReAct tool-lookup loop...")
            answer = react_pipeline(message, profile_state, client_id)
            answer = _check_answer(message, answer)
        else:
            _report(
                progress,
                0.5,
                "Step 3/4: Running ToT strategy analysis "
                "(3 analysts + critic + synthesis, ~30-60s)...",
            )
            answer, confidence_score = tot_pipeline(message, profile_state)
            answer = _check_answer(message, answer)
            if confidence_score is not None:
                answer = ConfidenceRouter.annotate(
                    answer, ConfidenceRouter.classify(confidence_score)
                )
        _report(progress, 0.9, "Step 4/4: Finalizing answer...")
    except RuntimeError as exc:
        logger.warning("respond: missing API key on follow-up: %s", exc)
        return str(exc), profile_state, "error"
    except Exception:
        logger.exception(
            "respond: follow-up pipeline failed (route=%s) for message=%r",
            route,
            message,
        )
        return (
            "That question hit an unexpected error from the LLM backend — "
            "this can happen intermittently. Try rephrasing or asking "
            "again.",
            profile_state,
            "error",
        )
    return answer, profile_state, route
