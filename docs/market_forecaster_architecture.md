# Market Forecaster — System Architecture

Aligned to the reference banking multi-agent pattern (coordinator/sub-agent, per-domain tool servers, observability, evaluation, session store), with Google added as the identity provider.

```mermaid
flowchart TD
    UI["Advisor UI<br/>(Gradio chat)"]
    AUTH_ID["Authentication<br/>Google (OAuth)"]
    API["API"]
    PII["PII<br/>Redaction"]
    AUTHZ["Authorisation<br/>(guardrails.py)"]
    EVAL["Agent Evaluation<br/>Suite (evaluators.py)"]

    subgraph PCREW["Profile Summary Crew<br/>(portfolio submission)"]
        AGG["Data Aggregator"]
        PROFANALYST["Portfolio Analyst"]
        AGG --> PROFANALYST
    end

    ALERTS["Portfolio Rollup<br/>Alerts (alerts.py)"]

    subgraph AGENTS[" "]
        COORD["Router Agent<br/>(Coordinator)"]
        REACT["ReAct<br/>Agent"]
        TOT["ToT<br/>Crew"]
        COORD --> REACT
        COORD --> TOT
    end

    CONTENT["ContentPolicyGuard<br/>(guardrails.py)"]
    CONF["ConfidenceRouter<br/>(ToT confidence flag)"]

    subgraph LLMS["LLM Providers"]
        SELF_LLM["Self Hosted<br/>LLM"]
        THIRD_LLM["Third-party<br/>LLM"]
    end

    subgraph OBS["Observability"]
        OBS1["Prompts, Agent<br/>calls, Tool Calls<br/>etc."]
        OBS2["CPU, Memory,<br/>Disk utilisation<br/>etc."]
    end

    SESSION[("Session Store<br/>(in-memory)")]
    SESSION_NOTE["Portfolio + profile summary<br/>ToT branch state"]
    SQLITE_DB[("clients.db<br/>SQLite · per-advisor")]

    YAHOO_MCP["Yahoo Finance<br/>Adapter"]
    AV_MCP["Alpha Vantage<br/>Adapter"]
    CHROMA_MCP["ChromaDB<br/>Retrieval"]
    SEC_EDGAR["SEC EDGAR<br/>10-K filings"]

    PRICE["get_price"]
    DIV["get_dividends"]
    SPLIT["get_splits"]
    RATING["get_ratings"]
    EARN["get_earnings"]
    NEWS["get_news_sentiment"]
    FILING_CHUNKS["search_filings<br/>(10-K chunks)"]
    HISTORY_CHUNKS["search_client_history<br/>(past turns)"]

    UI --> AUTH_ID
    UI --> API
    API --> PII
    PII --> PCREW
    PII --> COORD

    AGG --> YAHOO_MCP
    PCREW --> ALERTS
    SQLITE_DB -. prior snapshot .-> ALERTS
    ALERTS --> UI

    COORD <--> AUTHZ
    COORD --> EVAL
    EVAL --> SELF_LLM
    EVAL --> THIRD_LLM

    OBS <--> AGENTS

    REACT --> YAHOO_MCP
    REACT --> AV_MCP
    REACT --> CHROMA_MCP
    YAHOO_MCP -. reused for citations .-> TOT

    REACT --> CONTENT
    TOT --> CONTENT
    CONTENT --> CONF
    CONF --> UI

    AGENTS --> SESSION
    SESSION --- SESSION_NOTE
    SQLITE_DB -. load/save per advisor_id .-> SESSION

    YAHOO_MCP --> PRICE
    YAHOO_MCP --> DIV
    YAHOO_MCP --> SPLIT
    YAHOO_MCP --> RATING
    AV_MCP --> EARN
    AV_MCP --> NEWS
    SEC_EDGAR --> CHROMA_MCP
    CHROMA_MCP --> FILING_CHUNKS
    CHROMA_MCP --> HISTORY_CHUNKS

    classDef blue fill:#cfe2f3,stroke:#6fa8dc,color:#1c2833
    classDef green fill:#d9ead3,stroke:#93c47d,color:#1c2833
    classDef tan fill:#fce8b2,stroke:#e6b800,color:#1c2833
    classDef grey fill:#f2f2f2,stroke:#999999,color:#1c2833
    classDef store fill:#a4c2f4,stroke:#3d85c6,color:#1c2833
    classDef purple fill:#e0d3f2,stroke:#9370c7,color:#1c2833

    class UI,AUTH_ID,API,PII,AUTHZ blue
    class COORD,REACT,TOT,EVAL,YAHOO_MCP,AV_MCP,CHROMA_MCP,SEC_EDGAR,AGG,PROFANALYST green
    class SELF_LLM,THIRD_LLM tan
    class PRICE,DIV,SPLIT,RATING,EARN,NEWS,FILING_CHUNKS,HISTORY_CHUNKS,OBS1,OBS2 grey
    class SESSION,SQLITE_DB store
    class ALERTS,CONTENT,CONF purple
```

## Component Reference

### Entry & Security
| Component | Role |
|---|---|
| **Advisor UI (Gradio chat)** | Advisor-facing chat entry point — portfolio pasted in, questions asked here |
| **Authentication** | Google (OAuth) — verifies the advisor's identity before any session starts |
| **API** | Single entry point into the backend, sitting between the UI and the agent layer |
| **PII Redaction** | Strips personally identifiable information from requests before they reach any agent |
| **Authorisation** | Checked by the Router Agent before dispatching to any sub-agent — enforced by `guardrails.py`'s tool allow-lists and startup action-constraint audit |

### Portfolio Submission Path
| Component | Role |
|---|---|
| **Profile Summary Crew** | Runs instead of the question-routing path when the advisor pastes a portfolio (`looks_like_portfolio`) — two CrewAI agents in sequence |
| **Data Aggregator** | Fetches raw market data for every ticker via the Yahoo Finance Adapter, hands the raw JSON to the Portfolio Analyst |
| **Portfolio Analyst** | Turns the raw data into the plain-English profile summary shown to the advisor |
| **Portfolio Rollup Alerts** | `alerts.py` — diffs the newly fetched data against that same client's *previous* snapshot (read from clients.db) for rating changes and 5%+/10%+ price moves; silent if there's no prior snapshot or nothing changed, never fabricates a change |

### Agent Layer
| Component | Role |
|---|---|
| **Router Agent (Coordinator)** | Classifies each advisor follow-up question and routes it to the correct path; the only agent that talks to Authorisation and Agent Evaluation Suite directly |
| **ReAct Agent** | Handles straight/factual questions — fetch, then answer. The only agent with live tool access (Yahoo Finance, Alpha Vantage, ChromaDB) |
| **ToT Crew** | Handles complex/explanatory questions — 3 analyst lenses generate candidate reasoning, a critic scores them, a synthesizer produces the final answer. No tool calls of its own — cites news/earnings already fetched for the profile summary (dashed edge from Yahoo Finance Adapter) rather than making a fresh retrieval call |

### LLM & Evaluation
| Component | Role |
|---|---|
| **Agent Evaluation Suite** | `evaluators.py` — faithfulness, relevancy, and agent-task correctness checks; sits between the Router Agent and the LLM providers |
| **Self Hosted LLM** | In-house model option — not yet wired in |
| **Third-party LLM** | External model provider — the model currently in use |
| **ContentPolicyGuard** | `guardrails.py` — flags advice-like phrasing ("you should buy...") in every ReAct/ToT answer and prepends a disclaimer rather than blocking it |
| **ConfidenceRouter** | `guardrails.py` — reads the ToT Critic's own self-rated confidence score and flags the answer as low-confidence in the UI when it's weak; ReAct answers skip this (no critic score exists for that path) |

### Tooling Layer (per source)
| Component | Role |
|---|---|
| **Yahoo Finance Adapter** | Exposes live data tools to the ReAct Agent (and the Data Aggregator) → **get_price**, **get_dividends**, **get_splits**, **get_ratings** |
| **Alpha Vantage Adapter** | Exposes secondary-source tools to the ReAct Agent → **get_earnings**, **get_news_sentiment** |
| **ChromaDB Retrieval** | Exposes semantic search to the ReAct Agent only → **search_filings** (10-K chunks, backed by SEC EDGAR) and **search_client_history** (this client's past turns) |
| **SEC EDGAR** | Fetches a ticker's latest 10-K on first request; chunked and embedded into ChromaDB, reused by every later query for that ticker (any client, any session) |

### Cross-Cutting
| Component | Role |
|---|---|
| **Observability** | Two feeds: (1) prompts, agent calls, tool calls — LangSmith tracing; (2) CPU, memory, disk utilisation — not yet built. Wraps the whole agent layer. |
| **Session Store** | In-memory `clients_state` — holds the active portfolio, profile summary, and ToT branch state for the current browser session; the agent group reads/writes it directly |
| **clients.db** | SQLite, partitioned by `advisor_id` (composite `(advisor_id, id)` primary key) — the durable long-term memory behind Session Store. Survives app restarts and browser sessions; each advisor's Google sign-in only ever loads their own rows, never another advisor's |
