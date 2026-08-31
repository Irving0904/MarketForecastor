"""Guardrails for Market Forecaster's agents.

Covers the four boundary categories a multi-agent financial system needs:

  1. Data Access Controls - which tools each agent may call, validated once
     at agent-construction time against a project-wide allow-list.
  2. Action Constraints   - a startup audit that fails loudly if any agent
     is ever wired to a tool capable of a side effect. This app is
     read-only by design: it answers questions, it never trades or edits
     records.
  3. Content Policy       - flags direct investment-advice phrasing
     ("you should buy...") in generated text and prepends a disclaimer
     rather than presenting it as a recommendation.
  4. Confidence Routing   - surfaces the ToT Crew's own critic score to the
     advisor when a synthesized answer wasn't well-supported, instead of
     hiding a shaky analysis behind confident-sounding prose.

Guards 1 and 2 run at construction/startup and raise loudly on violation.
Guards 3 and 4 run on generated text and annotate rather than block -- the
underlying facts stay visible, they just aren't presented as more certain
or more prescriptive than they actually are.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data Access Controls
# ---------------------------------------------------------------------------

class DataAccessViolation(Exception):
    """Raised when an agent is constructed with a tool outside its allow-list."""


class DataAccessGuard:
    """
    Restricts what a given agent is allowed to call. Construct one at the
    point each agent is built (profile_crew.py, tot_crew.py, react_agent.py)
    -- validation happens in __init__, so a bad grant fails immediately at
    startup rather than surfacing as a confusing runtime error later.

    Allow-list based, not deny-list: a new tool must be explicitly added to
    KNOWN_SAFE_TOOLS before any agent can be granted it.
    """

    # Every tool any agent in this project can call, named exactly as
    # CrewAI (string tool names) or LangChain (@tool function names)
    # expose them. Extend this when a new tool is added.
    KNOWN_SAFE_TOOLS = {
        "Fetch Market Data",       # crewai tool -- profile_crew.py aggregator
        "get_holding_data",        # langchain tool -- react_agent.py
        "search_filings",          # langchain tool -- react_agent.py
        "search_client_history",   # langchain tool -- react_agent.py
    }

    def __init__(self, agent_name: str, allowed_tools: Iterable[str]):
        self.agent_name = agent_name
        self.allowed_tools = set(allowed_tools)

        unknown = self.allowed_tools - self.KNOWN_SAFE_TOOLS
        if unknown:
            raise DataAccessViolation(
                f"Agent '{agent_name}' was granted unrecognized tool(s): "
                f"{sorted(unknown)}. Add them to "
                f"DataAccessGuard.KNOWN_SAFE_TOOLS only after reviewing "
                f"exactly what they let the agent read."
            )
        logger.info(
            "DataAccessGuard: %s granted %s",
            agent_name,
            sorted(self.allowed_tools) or "no tools (pure reasoning)",
        )


# ---------------------------------------------------------------------------
# 2. Action Constraints
# ---------------------------------------------------------------------------

# Tool names are snake_case / space-separated (e.g. "Fetch Market Data",
# "search_client_history"), so a naive \b-bounded regex is NOT safe here --
# \b treats underscores as word characters, meaning "execute_trade" would
# NOT match \bexecute\b (no boundary before the trailing "_trade"). Split
# on non-alphabetic characters and check whole tokens instead.
_FORBIDDEN_ACTION_WORDS = {
    "write", "delete", "drop", "update",
    "execute", "trade", "buy", "sell",
    "transfer", "send", "post",
}
_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z]+")


def _contains_forbidden_action(tool_name: str) -> bool:
    tokens = _TOKEN_SPLIT_RE.split(tool_name.lower())
    return any(token in _FORBIDDEN_ACTION_WORDS for token in tokens if token)


class ActionConstraintViolation(Exception):
    """Raised at startup if any agent is wired to a write/execute-capable tool."""


# The complete tool grant for every agent in the app, listed here (not
# introspected from live crew objects) so this audit can run at import
# time, before any crew, LLM client, or API key is even touched.
AGENT_TOOL_REGISTRY: dict[str, list[str]] = {
    "profile_crew.Market Data Aggregator": ["Fetch Market Data"],
    "profile_crew.Portfolio Analyst": [],
    "tot_crew.Investment Analyst": [],
    "tot_crew.Risk Critic": [],
    "tot_crew.Lead Advisor": [],
    "react_agent.ReAct portfolio assistant": [
        "get_holding_data", "search_filings", "search_client_history",
    ],
    "router.Router": [],
}


class ActionConstraintGuard:
    """Startup-time check: Market Forecaster is read-only by design. This
    scans every agent's registered tool grant and fails loudly if anything
    looks write/execute-capable, so an accidental future addition of a
    dangerous tool stops the app from starting rather than reaching a
    live advisor session."""

    @staticmethod
    def audit(agent_tool_registry: dict[str, Iterable[str]] | None = None) -> None:
        registry = AGENT_TOOL_REGISTRY if agent_tool_registry is None else agent_tool_registry
        violations = [
            f"{agent} -> {tool}"
            for agent, tools in registry.items()
            for tool in tools
            if _contains_forbidden_action(tool)
        ]
        if violations:
            raise ActionConstraintViolation(
                "Write/execute-capable tool(s) detected -- Market Forecaster "
                "must remain read-only:\n  " + "\n  ".join(violations)
            )
        logger.info(
            "ActionConstraintGuard: startup audit passed, %d agents checked, all read-only",
            len(registry),
        )


def startup_checks() -> None:
    """Call once, before the Gradio server accepts any input (see main.py)."""
    ActionConstraintGuard.audit()


# ---------------------------------------------------------------------------
# 3. Content Policy
# ---------------------------------------------------------------------------

# Phrasing patterns that read as a direct, personal investment
# recommendation rather than a factual/explanatory statement. Deliberately
# conservative -- a false positive (flagging something borderline) is
# cheaper than a false negative (a recommendation slipping through
# unflagged from an app with no licensed advisor behind it).
_ADVICE_PATTERNS = [
    r"\byou should (buy|sell|hold|invest|short)\b",
    r"\bi recommend (buying|selling|holding|investing)\b",
    r"\bnow is the time to (buy|sell)\b",
    r"\bthis is a (strong|great|solid) buy\b",
    r"\byou('d| would) be (wise|smart) to (buy|sell)\b",
]
_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), re.IGNORECASE)


@dataclass
class ContentPolicyResult:
    flagged: bool
    matched_phrases: list[str] = field(default_factory=list)


class ContentPolicyGuard:
    """Post-generation scorer, run centrally in core/orchestrator.py on
    every answer shown to the advisor (both the ReAct and ToT paths).
    Flags language that reads as a direct investment recommendation.

    Starts as a simple pattern match -- cheap, deterministic, zero added
    latency. If the false-positive rate turns out too high, route flagged
    text through an LLM classifier instead of (not in addition to) the
    regex pass, to keep the common case cheap.
    """

    @staticmethod
    def check(answer_text: str) -> ContentPolicyResult:
        matches = _ADVICE_RE.findall(answer_text)
        flat = [m if isinstance(m, str) else next(g for g in m if g) for m in matches]
        return ContentPolicyResult(flagged=bool(matches), matched_phrases=flat)

    @staticmethod
    def annotate(answer_text: str, result: ContentPolicyResult) -> str:
        """If flagged, prepend a disclaimer rather than silently blocking --
        the underlying facts may still be useful, this just makes sure they
        aren't presented as a personal recommendation."""
        if not result.flagged:
            return answer_text
        disclaimer = (
            "[This response was flagged for advice-like phrasing -- treat "
            "it as factual/explanatory context only, not a personal "
            "investment recommendation.]\n\n"
        )
        return disclaimer + answer_text


# ---------------------------------------------------------------------------
# 4. Confidence-Based Routing
# ---------------------------------------------------------------------------

_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def critic_output_guardrail(output):
    """CrewAI native Task guardrail (see tot_crew.py's critic_task,
    `Task(guardrail=critic_output_guardrail)`).

    ConfidenceRouter needs a numeric score parsed out of the Risk Critic's
    free-text critique. This was originally written to return (False, ...)
    and force a retry when the required CONFIDENCE line was missing --
    but in practice (tested against the live Anthropic backend) the
    critic doesn't reliably add it even when told exactly what's wrong,
    and CrewAI's guardrail contract *raises* once retries are exhausted,
    which took down the entire ToT answer over a missing confidence badge
    on a backend that already flakes intermittently on its own. A missing
    self-assessment score isn't worth that: always accept, and let
    ConfidenceRouter.extract_score() return None downstream, which
    core/orchestrator.py already handles by simply skipping the
    confidence annotation for that turn.

    Deliberately left without a `-> tuple[bool, str]` return annotation:
    this module uses `from __future__ import annotations`, which makes
    all annotations lazy strings at runtime (PEP 563). CrewAI's Task
    validator inspects the *live* return annotation with
    `typing.get_origin`/`get_args` to confirm a guardrail's shape, and a
    stringified annotation fails that check silently as "malformed",
    raising a pydantic ValidationError at Task construction. No
    annotation here means CrewAI skips that check entirely.
    """
    if not _CONFIDENCE_RE.search(output.raw):
        logger.warning(
            "critic_output_guardrail: critic omitted the CONFIDENCE line, "
            "ConfidenceRouter will have no score for this turn"
        )
    return True, output.raw


@dataclass
class ConfidenceResult:
    level: str  # "high" | "medium" | "low"
    score: float
    should_flag_ui: bool


class ConfidenceRouter:
    """Reads the ToT Crew's own Risk Critic score (0-10, extracted from
    critic_task's output by tot_pipeline) and classifies it. Thresholds
    are a starting point -- tune against real ToT Crew runs once you have
    a larger sample to look at."""

    HIGH_CONFIDENCE_THRESHOLD = 7.5
    LOW_CONFIDENCE_THRESHOLD = 4.5

    @staticmethod
    def extract_score(critic_output_raw: str) -> float | None:
        match = _CONFIDENCE_RE.search(critic_output_raw)
        if not match:
            return None
        return max(0.0, min(10.0, float(match.group(1))))

    @classmethod
    def classify(cls, score: float) -> ConfidenceResult:
        if score >= cls.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceResult("high", score, should_flag_ui=False)
        if score >= cls.LOW_CONFIDENCE_THRESHOLD:
            return ConfidenceResult("medium", score, should_flag_ui=False)
        return ConfidenceResult("low", score, should_flag_ui=True)

    @staticmethod
    def annotate(answer_text: str, result: ConfidenceResult) -> str:
        if not result.should_flag_ui:
            return answer_text
        flag = (
            f"[Low confidence answer (critic score: {result.score:.0f}/10) "
            "-- the analysts' reasoning here was thin or weakly supported "
            "for your specific portfolio. Treat this as a starting point, "
            "not a definitive answer.]\n\n"
        )
        return flag + answer_text
