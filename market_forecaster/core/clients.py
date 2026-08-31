"""Per-client session management.

Each client gets its own portfolio profile, trace log, and chat history,
keyed by a short id, so the advisor can switch between client profiles
without losing any one client's context. In-memory only — sessions reset
when the app restarts.
"""

import uuid

EMPTY_CLIENT = {"profile": {}, "trace": [], "chat": []}


def new_client_id() -> str:
    return uuid.uuid4().hex[:8]


def empty_client(name: str) -> dict:
    return {"name": name, **EMPTY_CLIENT}


def radio_choices(clients: dict) -> list[tuple[str, str]]:
    """(display label, client id) pairs for the client-switcher Radio."""
    return [(client["name"], client_id) for client_id, client in clients.items()]


def name_exists(name: str, clients: dict) -> bool:
    """Case-insensitive check — client names must be unique."""
    name = name.strip().lower()
    return any(c["name"].strip().lower() == name for c in clients.values())
