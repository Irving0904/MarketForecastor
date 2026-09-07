"""LangChain router chain: classifies a follow-up question as a quick
"straight" lookup or a "tot" (Tree-of-Thought) deliberation."""

import logging

from langchain_core.prompts import ChatPromptTemplate

from market_forecaster.agents.utils import extract_text
from market_forecaster.config import get_chat_model

logger = logging.getLogger(__name__)

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the advisor's question as either 'straight' or "
            "'tot'.\n"
            "'straight': answerable by looking up one or two concrete "
            "facts about a single holding (price, yield, a specific "
            "rating, an isolated 10-K/news fact) with no interpretation "
            "needed.\n"
            "'tot': anything requiring interpretation, synthesis across "
            "multiple factors, or judgment — tradeoffs, comparisons, "
            "strategy (rebalancing, risk exposure, hypothetical "
            "scenarios), open-ended or causal questions (e.g. 'why did "
            "X move', 'what do you think about...', 'should I be "
            "worried about...'), or anything without one obvious "
            "factual answer.\n"
            "Reply with exactly one word: straight or tot.",
        ),
        ("human", "Profile: {profile}\n\nQuestion: {question}"),
    ]
)


def router_agent(message: str, profile_summary: str) -> str:
    logger.info("router_agent: classifying message=%r", message)
    llm = get_chat_model(max_tokens=10)
    chain = ROUTER_PROMPT | llm
    response = chain.invoke({"profile": profile_summary, "question": message})
    label = extract_text(response.content).strip().lower()
    # Default *toward* tot on an ambiguous/empty label — open-ended
    # questions should get deliberation, not the cheap path by accident.
    route = "straight" if "straight" in label else "tot"
    logger.info("router_agent: raw_label=%r -> route=%r", label, route)
    return route


CONSTRUCTION_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify whether the advisor's question is asking to BUILD "
            "or RECOMMEND a brand-new portfolio allocation (as opposed to "
            "analyzing, explaining, or asking about an existing holding). "
            "Reply with exactly one word:\n"
            "'equity' — asks to build/recommend a portfolio using only "
            "stocks/equities and/or ETFs, with no mention of also wanting "
            "bonds, fixed income, or mutual funds.\n"
            "'diversified' — asks to build/recommend a portfolio that "
            "explicitly spans multiple asset classes (mentions "
            "diversification, bonds/fixed income, or mutual funds "
            "alongside stocks).\n"
            "'none' — anything else: analyzing an existing holding, a "
            "factual lookup, a why-did-this-move question, or any "
            "question that isn't asking to construct a new portfolio.",
        ),
        ("human", "Question: {question}"),
    ]
)


def construction_router(message: str) -> str:
    """Classifies a message as a request to construct a new portfolio
    ('equity' or 'diversified') or not ('none') -- checked in
    orchestrator.py before the existing paste-a-portfolio / straight-vs-tot
    routing, so it applies whether or not a client is currently loaded."""
    logger.info("construction_router: classifying message=%r", message)
    llm = get_chat_model(max_tokens=10)
    chain = CONSTRUCTION_ROUTER_PROMPT | llm
    response = chain.invoke({"question": message})
    label = extract_text(response.content).strip().lower()
    if "equity" in label:
        route = "equity"
    elif "diversif" in label:
        route = "diversified"
    else:
        route = "none"
    logger.info("construction_router: raw_label=%r -> route=%r", label, route)
    return route
