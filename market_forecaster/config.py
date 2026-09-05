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


def get_google_oauth_credentials() -> tuple[str, str]:
    """Google Cloud Console OAuth 2.0 Client ID (Web application type) --
    see README Setup for how to create one. Checked at auth.py's
    module-setup time, so a missing credential fails before the server
    binds to a port, same as the other required-secret checks here."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID and/or GOOGLE_CLIENT_SECRET is not set. Set "
            "both in your environment (or .env file) and restart the app "
            "to enable Google sign-in. See README Setup for how to create "
            "an OAuth 2.0 Client ID in Google Cloud Console."
        )
    return client_id, client_secret


def get_alpha_vantage_api_key() -> str | None:
    """Optional -- Alpha Vantage is a secondary data source used only to
    cross-check Yahoo Finance's earnings/news for the tickers in a
    submitted portfolio, never a hard requirement. A missing key just
    means that cross-check is silently skipped (see data/alpha_vantage.py),
    unlike the required-secret checks above which fail app startup."""
    return os.environ.get("ALPHA_VANTAGE_API_KEY")


def get_session_secret() -> str:
    """Signs the session cookie that holds an advisor's identity between
    requests. Required, not defaulted: a missing or predictable secret
    would let a session cookie be forged."""
    secret = os.environ.get("SESSION_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "SESSION_SECRET_KEY is not set. Generate one with "
            "`python -c \"import secrets; print(secrets.token_hex(32))\"` "
            "and set it in your environment (or .env file), then restart "
            "the app."
        )
    return secret
