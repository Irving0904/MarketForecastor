"""Evaluators for Market Forecaster's generated content.

Guardrails (guardrails.py) *act* on output -- they annotate or force a
retry. Evaluators *measure* it -- they produce a score that gets logged
for every request, giving a real signal to look at across a session
instead of only reacting to the worst cases.

FaithfulnessEvaluator and RelevancyEvaluator are QAG-style
(Question-Answer-Generation): rather than asking an LLM for one holistic
subjective "how good is this, 1-10" score, each decomposes the text into
short, atomic, checkable claims and asks for a YES/NO/IDK verdict per
claim against a source, then reports the claim-level pass rate. That's
far more auditable than a single number, and it's what actually catches
CrewAI's occasional confident-but-ungrounded synthesis.

Both use one confined LLM call per evaluation (decompose + verify
combined) rather than one call per claim -- a deliberate cost/latency
tradeoff for a per-request check; splitting into separate calls would be
more rigorous but roughly triples the LLM calls on every profile build or
answer.

AgentTaskEvaluator is different in kind: a deterministic, offline
exact-match check of the router chain against a hand-labeled test set.
Run it directly (`python -m market_forecaster.evaluators`) after touching
router.py's prompt to catch a regression before it reaches a real
session -- it is not part of the live request path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from market_forecaster.agents.utils import extract_text
from market_forecaster.config import get_chat_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QAG-style claim checking (Faithfulness, Relevancy)
# ---------------------------------------------------------------------------

@dataclass
class ClaimCheck:
    claim: str
    verdict: str  # "YES" | "NO" | "IDK"


@dataclass
class QAGResult:
    claims: list[ClaimCheck] = field(default_factory=list)
    pass_rate: float = 1.0
    passed: bool = True


def _parse_qag_json(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    candidate = match.group(0) if match else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # A response that hit its max_tokens cap mid-array leaves the last
    # claim's object incomplete and the array unclosed -- json.loads on
    # the whole thing fails even though every claim before the cutoff is
    # still valid JSON. Salvage those instead of discarding every claim
    # for one truncated tail (seen in practice on longer profile
    # summaries, which decompose into more atomic claims).
    objects = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    salvaged = []
    for obj_str in objects:
        try:
            salvaged.append(json.loads(obj_str))
        except json.JSONDecodeError:
            continue
    if salvaged:
        logger.warning(
            "QAG evaluator: response was truncated, salvaged %d/%d claim object(s)",
            len(salvaged),
            len(objects),
        )
    else:
        logger.warning("QAG evaluator: could not parse JSON from LLM output: %r", raw[:200])
    return salvaged


def _run_qag(system_prompt: str, source: str, text: str, threshold: float) -> QAGResult:
    # Generous cap: a longer summary/answer decomposes into more atomic
    # claims, and a truncated response used to be silently treated as a
    # total parse failure. _parse_qag_json now salvages whatever complete
    # claim objects it can from a truncated tail as a backstop, but a
    # bigger cap means that backstop rarely has to do any work.
    llm = get_chat_model(max_tokens=3000)
    prompt = (
        f"{system_prompt}\n\n"
        f"SOURCE:\n{source}\n\nTEXT TO CHECK:\n{text}\n\n"
        'Respond with ONLY a JSON array like '
        '[{"claim": "...", "verdict": "YES"}, ...]. No other text.'
    )
    response = llm.invoke(prompt)
    raw = extract_text(response.content)
    items = _parse_qag_json(raw)
    if not items:
        # Fail open: an unparseable evaluator response is an evaluator bug,
        # not evidence the underlying answer is bad -- log it and let the
        # real answer through rather than blocking on our own parsing miss.
        logger.warning("QAG evaluator: no claims parsed from response, failing open")
        return QAGResult(claims=[], pass_rate=1.0, passed=True)

    claims = [
        ClaimCheck(str(i.get("claim", "")), str(i.get("verdict", "IDK")).upper())
        for i in items
    ]
    # IDK means the source doesn't say enough to confirm or deny -- that's
    # not the same as NO (contradicted / genuinely unsupported). A
    # portfolio summary or advisor answer is expected to contain
    # analytical/interpretive claims ("this makes the holding
    # growth-oriented") that no source JSON will ever literally confirm;
    # penalizing those as failures the same as an outright NO produced a
    # real false-positive pattern in testing (summaries with zero
    # contradicted facts still scoring 0.6-0.8). Only an explicit NO
    # counts against the score.
    no = sum(1 for c in claims if c.verdict == "NO")
    pass_rate = 1.0 - (no / len(claims))
    return QAGResult(claims=claims, pass_rate=pass_rate, passed=pass_rate >= threshold)


class FaithfulnessEvaluator:
    """Checks that a generated profile summary doesn't state facts
    unsupported by the raw market data it was supposed to be built from."""

    THRESHOLD = 0.8

    @classmethod
    def check(cls, generated_text: str, source_data: str) -> QAGResult:
        return _run_qag(
            "Decompose TEXT TO CHECK into short, atomic factual claims. "
            "For each claim, answer YES if it is directly supported by "
            "SOURCE, NO if it contradicts or is not backed by SOURCE, or "
            "IDK if SOURCE does not say enough to tell either way.",
            source_data,
            generated_text,
            cls.THRESHOLD,
        )


class RelevancyEvaluator:
    """Checks that a generated answer's claims actually address the
    question asked, rather than drifting onto tangential facts.

    Deliberately scoped to catch topic drift, not to demand every claim be
    the single literal fact asked for -- a financial-advisor answer that
    adds directly-supporting context (e.g. the analyst rating alongside a
    requested price) is exactly the kind of answer this app should give,
    and an evaluator that flags that as "irrelevant" trains advisors to
    ignore its warnings."""

    THRESHOLD = 0.6

    @classmethod
    def check(cls, question: str, generated_text: str) -> QAGResult:
        return _run_qag(
            "Decompose TEXT TO CHECK (an answer from a portfolio advisor "
            "tool) into short, atomic claims. For each claim, answer YES "
            "if it helps answer SOURCE (the client's question) -- this "
            "includes the direct fact requested AND closely-related "
            "supporting context a financial advisor would reasonably "
            "include alongside it (e.g. an analyst rating or trend "
            "alongside a requested price). Answer NO only for claims that "
            "are genuinely off-topic or unrelated to the question, or IDK "
            "if unclear.",
            question,
            generated_text,
            cls.THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Deterministic agent-task evaluation (Router)
# ---------------------------------------------------------------------------

@dataclass
class RouterTestCase:
    question: str
    profile_summary: str
    expected_route: str  # "straight" | "tot"


# Drawn from real cases observed during manual testing of this app,
# including the router-broadening regression check documented in the
# README's audit trail (the "why did MRNA stock go down" case).
ROUTER_TEST_SET: list[RouterTestCase] = [
    RouterTestCase("What is AAPL trading at right now?", "Holds AAPL, MSFT.", "straight"),
    RouterTestCase("What's TSLA's dividend yield?", "Holds TSLA.", "straight"),
    RouterTestCase("What is NVDA's trailing PE ratio?", "Holds NVDA.", "straight"),
    RouterTestCase("What is MSFT's analyst rating?", "Holds MSFT, AAPL.", "straight"),
    RouterTestCase("Why did MRNA stock go down?", "Holds MRNA.", "tot"),
    RouterTestCase(
        "Should I rebalance given my sector concentration?",
        "Holds NVDA, MSFT, GOOGL -- all tech.",
        "tot",
    ),
    RouterTestCase(
        "What do you think about my exposure to tech right now?",
        "Holds NVDA, MSFT, AAPL.",
        "tot",
    ),
    RouterTestCase(
        "Should I be worried about a downturn hitting my portfolio?",
        "Holds SPY, QQQ.",
        "tot",
    ),
]


@dataclass
class ReactToolTestCase:
    question: str
    session_state: dict
    client_id: str | None
    expected_tool: str  # "get_holding_data" | "search_filings" | "search_client_history"


# Minimal synthetic session state -- reused across cases. Real fetched
# market data isn't needed here: this checks which tool the agent chooses
# to call, not what it does with the tool's result, so a lightweight
# stand-in keeps this cheap and fast to run repeatedly.
_SAMPLE_SESSION_STATE = {
    "tickers": ["AAPL", "MSFT", "NVDA"],
    "raw_data": {
        "AAPL": {"price": 319.7, "sector": "Technology", "dividend_yield_pct": 0.34},
        "MSFT": {"price": 500.0, "sector": "Technology", "dividend_yield_pct": 0.71},
        "NVDA": {"price": 180.0, "sector": "Technology", "dividend_yield_pct": 0.03},
    },
}

REACT_TOOL_TEST_SET: list[ReactToolTestCase] = [
    ReactToolTestCase(
        "What is AAPL trading at right now?",
        _SAMPLE_SESSION_STATE,
        None,
        "get_holding_data",
    ),
    ReactToolTestCase(
        "What's MSFT's dividend yield?",
        _SAMPLE_SESSION_STATE,
        None,
        "get_holding_data",
    ),
    ReactToolTestCase(
        "What does NVDA's 10-K say about its business risk factors?",
        _SAMPLE_SESSION_STATE,
        None,
        "search_filings",
    ),
    ReactToolTestCase(
        "What did we discuss about AAPL the last time we talked?",
        _SAMPLE_SESSION_STATE,
        "sample_client_id",
        "search_client_history",
    ),
]


class AgentTaskEvaluator:
    """Deterministic exact-match evaluation of an agent's classification
    or tool-selection output against a hand-labeled test set."""

    @staticmethod
    def evaluate_router(test_set: list[RouterTestCase] | None = None) -> dict:
        from market_forecaster.agents.router import router_agent

        cases = test_set if test_set is not None else ROUTER_TEST_SET
        results = []
        for case in cases:
            actual = router_agent(case.question, case.profile_summary)
            results.append(
                {
                    "question": case.question,
                    "expected": case.expected_route,
                    "actual": actual,
                    "correct": actual == case.expected_route,
                }
            )
        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"accuracy": accuracy, "results": results}

    @staticmethod
    def evaluate_react_tools(test_set: list[ReactToolTestCase] | None = None) -> dict:
        """Checks which tool the ReAct agent calls first for a given
        question against a hand-labeled expectation -- ground truth exists
        (only one tool is the right one for each case), so this is
        exact-match, not LLM-judged, same as evaluate_router."""
        from market_forecaster.agents.react_agent import build_react_agent

        cases = test_set if test_set is not None else REACT_TOOL_TEST_SET
        results = []
        for case in cases:
            agent = build_react_agent(case.session_state, case.client_id)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": case.question}]}
            )
            actual_tool = None
            actual_args = None
            for m in result["messages"]:
                calls = getattr(m, "tool_calls", None)
                if calls:
                    actual_tool = calls[0]["name"]
                    actual_args = calls[0]["args"]
                    break
            results.append(
                {
                    "question": case.question,
                    "expected_tool": case.expected_tool,
                    "actual_tool": actual_tool,
                    "actual_args": actual_args,
                    "correct": actual_tool == case.expected_tool,
                }
            )
        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"accuracy": accuracy, "results": results}


# ---------------------------------------------------------------------------
# Calibration (ToT confidence vs. independent quality signal)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationTestCase:
    question: str
    session_state: dict  # must include "summary" (tot_pipeline's only requirement)


# A hardcoded summary rather than a fresh yfinance + Profile Summary Crew
# run: calibration already costs a full ToT pipeline (5 LLM calls) plus one
# RelevancyEvaluator call per case, and this keeps that the only variable
# cost instead of also depending on live market data / another crew run.
_CALIBRATION_SESSION_STATE = {
    "summary": (
        "This portfolio is heavily concentrated in two mega-cap Technology "
        "names, AAPL and MSFT, with a combined market cap over $8 trillion "
        "and no sector diversification. Dividend income is minimal (0.34% "
        "AAPL, 0.71% MSFT), and both trade at rich valuations (AAPL 36.6x "
        "trailing earnings, MSFT 28.2x trailing earnings)."
    ),
    "tickers": ["AAPL", "MSFT"],
}

CALIBRATION_TEST_SET: list[CalibrationTestCase] = [
    CalibrationTestCase(
        "Should I rebalance given my concentration in tech?",
        _CALIBRATION_SESSION_STATE,
    ),
    CalibrationTestCase(
        "What do you think about my exposure to AI-driven growth right now?",
        _CALIBRATION_SESSION_STATE,
    ),
    CalibrationTestCase(
        "Should I be worried about a pullback given how richly valued these are?",
        _CALIBRATION_SESSION_STATE,
    ),
]


class CalibrationEvaluator:
    """Not a pass/fail test -- there's no ground truth for "was this ToT
    answer actually well-supported," so this can't be exact-match like
    AgentTaskEvaluator. Instead it cross-references two independent
    signals on the same answer: the ToT Risk Critic's own self-rated
    CONFIDENCE score (via ConfidenceRouter) and RelevancyEvaluator's
    QAG-based score. If "low confidence" is a meaningful signal, answers
    the critic flagged low should tend to score lower on relevancy too;
    if the two are uncorrelated, the critic's self-rating isn't tracking
    anything real and the threshold values in ConfidenceRouter need
    revisiting.

    Expensive and, per this project's documented CrewAI flakiness, prone
    to occasional per-case failures -- keep the test set small, and a
    failed case is skipped (logged) rather than aborting the whole run."""

    @staticmethod
    def run(test_set: list[CalibrationTestCase] | None = None) -> dict:
        from market_forecaster.agents.tot_crew import tot_pipeline
        from market_forecaster.guardrails import ConfidenceRouter

        cases = test_set if test_set is not None else CALIBRATION_TEST_SET
        results = []
        for case in cases:
            try:
                answer, confidence_score = tot_pipeline(case.question, case.session_state)
            except Exception:
                logger.warning(
                    "CalibrationEvaluator: tot_pipeline failed for question=%r, skipping",
                    case.question,
                    exc_info=True,
                )
                continue
            level = (
                ConfidenceRouter.classify(confidence_score).level
                if confidence_score is not None
                else "unknown"
            )
            relevancy = RelevancyEvaluator.check(case.question, answer)
            results.append(
                {
                    "question": case.question,
                    "confidence_score": confidence_score,
                    "confidence_level": level,
                    "relevancy_pass_rate": relevancy.pass_rate,
                }
            )
        return {"results": results, "bucket_avg_relevancy": CalibrationEvaluator._bucket_avg(results)}

    @staticmethod
    def _bucket_avg(results: list[dict]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for r in results:
            buckets.setdefault(r["confidence_level"], []).append(r["relevancy_pass_rate"])
        return {level: sum(scores) / len(scores) for level, scores in buckets.items()}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)

    print("=== Router classification ===")
    router_report = AgentTaskEvaluator.evaluate_router()
    for r in router_report["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        print(f"[{mark}] expected={r['expected']!r} actual={r['actual']!r} question={r['question']!r}")
    router_correct = sum(r["correct"] for r in router_report["results"])
    router_total = len(router_report["results"])
    print(f"Accuracy: {router_report['accuracy']:.0%} ({router_correct}/{router_total})")

    print("\n=== ReAct tool selection ===")
    tool_report = AgentTaskEvaluator.evaluate_react_tools()
    for r in tool_report["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        print(
            f"[{mark}] expected={r['expected_tool']!r} actual={r['actual_tool']!r} "
            f"args={r['actual_args']!r} question={r['question']!r}"
        )
    tool_correct = sum(r["correct"] for r in tool_report["results"])
    tool_total = len(tool_report["results"])
    print(f"Accuracy: {tool_report['accuracy']:.0%} ({tool_correct}/{tool_total})")

    # Opt-in: a full ToT pipeline per case (5 LLM calls) makes this far
    # more expensive and slower than the two exact-match evals above, and
    # it's a directional signal to read, not a pass/fail gate -- running
    # it by default on every invocation would bury the cheap, fast checks
    # under noise and cost most callers don't want.
    if "--calibration" in sys.argv:
        print("\n=== Confidence calibration (ToT critic score vs. relevancy) ===")
        calibration_report = CalibrationEvaluator.run()
        for r in calibration_report["results"]:
            print(
                f"confidence={r['confidence_level']:<8} "
                f"(score={r['confidence_score']}) "
                f"relevancy_pass_rate={r['relevancy_pass_rate']:.2f}  "
                f"question={r['question']!r}"
            )
        print("\nAverage relevancy pass_rate by confidence bucket:")
        for level, avg in calibration_report["bucket_avg_relevancy"].items():
            print(f"  {level:<8} {avg:.2f}")
        print(
            "\n(Directional only: 'low' should average lower than 'high' if "
            "the critic's self-rating tracks real quality. Small sample, "
            "expect noise -- not a pass/fail result.)"
        )
    else:
        print("\n(Skipping confidence calibration -- expensive, opt in with --calibration)")
