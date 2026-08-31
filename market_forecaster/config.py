"""Environment loading and LLM client factories.

Note: the Anthropic SDK pinned in this environment rejects `temperature`
for this model ("`temperature` is deprecated for this model"), so no
call anywhere in this project passes it — LangChain/CrewAI both default
to omitting it as long as it's never set explicitly.

.env is actually loaded in market_forecaster/__init__.py, not here — see
that file's docstring for why it has to happen there.
"""

import os

from crewai import LLM
from langchain_anthropic import ChatAnthropic

MODEL = "claude-sonnet-5"


def require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in your environment "
            "(or .env file) and restart the app to enable the advisor."
        )


def get_chat_model(max_tokens: int = 500) -> ChatAnthropic:
    """LangChain-compatible Claude client, for the router and ReAct agent."""
    require_api_key()
    return ChatAnthropic(model=MODEL, max_tokens=max_tokens)


def get_crew_llm(max_tokens: int = 600) -> LLM:
    """CrewAI-compatible Claude client, for the Profile Summary and ToT crews."""
    require_api_key()
    return LLM(model=f"anthropic/{MODEL}", max_tokens=max_tokens)


def get_sec_user_agent() -> str:
    """SEC EDGAR requires every request to identify a real contact
    (https://www.sec.gov/os/webmaster-faq#developers) — requests without
    one get rate-limited or blocked."""
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT is not set. Set it in your environment "
            "(or .env file) to something like 'YourApp your@email.com' "
            "and restart the app to enable SEC filing search."
        )
    return user_agent
