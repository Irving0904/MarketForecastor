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
