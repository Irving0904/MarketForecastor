"""SQLite persistence for client sessions — long-term memory across app
restarts. `clients_state` in ui/app.py stays the in-memory source of truth
during a session; this module is what makes it durable.

Every row is partitioned by `advisor_id` (the Google account's stable `sub`
claim — see market_forecaster/auth.py). The primary key is the *pair*
`(advisor_id, id)`, not `id` alone: `id` is a locally-generated
`uuid4().hex[:8]` with no cross-advisor uniqueness guarantee, so keying on
the pair means a write or delete scoped to the wrong advisor matches zero
rows at the SQL level -- an actual isolation guarantee, not just a filter
applied after the fact. Every function here takes `advisor_id` as its first
argument; never construct a query against this table elsewhere.

DB file lives at the project root (sibling to .env, main.py) and is
gitignored, same reasoning as .env: it holds client PII (names, holdings).
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "clients.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the clients table if it doesn't exist. Call once at startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clients (
                id TEXT NOT NULL,
                advisor_id TEXT NOT NULL,
                name TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                chat_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (advisor_id, id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clients_advisor ON clients(advisor_id)"
        )
    logger.info("client_store: initialized database at %s", DB_PATH)


def save_client(advisor_id: str, client_id: str, client: dict) -> None:
    """Upsert one client's full session state, scoped to advisor_id."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO clients (id, advisor_id, name, profile_json, trace_json, chat_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(advisor_id, id) DO UPDATE SET
                name=excluded.name,
                profile_json=excluded.profile_json,
                trace_json=excluded.trace_json,
                chat_json=excluded.chat_json,
                updated_at=excluded.updated_at
            """,
            (
                client_id,
                advisor_id,
                client.get("name", ""),
                json.dumps(client.get("profile", {}), default=str),
                json.dumps(client.get("trace", []), default=str),
                json.dumps(client.get("chat", []), default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    logger.debug(
        "client_store: saved client %s (%s) for advisor %s",
        client_id,
        client.get("name"),
        advisor_id,
    )


def load_all_clients(advisor_id: str) -> dict[str, dict]:
    """Read every client persisted for this advisor into the shape
    clients_state expects. Never returns another advisor's rows."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, profile_json, trace_json, chat_json FROM clients "
            "WHERE advisor_id = ?",
            (advisor_id,),
        ).fetchall()
    clients = {}
    for client_id, name, profile_json, trace_json, chat_json in rows:
        clients[client_id] = {
            "name": name,
            "profile": json.loads(profile_json),
            "trace": json.loads(trace_json),
            "chat": json.loads(chat_json),
        }
    logger.info(
        "client_store: loaded %d client(s) from disk for advisor %s",
        len(clients),
        advisor_id,
    )
    return clients


def search_clients(advisor_id: str, query: str) -> list[tuple[str, str]]:
    """(name, id) pairs for this advisor's clients whose name contains
    query, case-insensitive. No current caller in the codebase -- scoped
    anyway so it isn't a trap for whoever wires it up next."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, id FROM clients WHERE advisor_id = ? AND name LIKE ? "
            "ORDER BY name",
            (advisor_id, f"%{query}%"),
        ).fetchall()
    return [(name, client_id) for name, client_id in rows]


def delete_client(advisor_id: str, client_id: str) -> None:
    """Deletes only if client_id actually belongs to advisor_id -- a
    client_id from a different advisor (guessed, replayed, or stale UI
    state) matches nothing and this is a silent no-op, not an error."""
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM clients WHERE advisor_id = ? AND id = ?",
            (advisor_id, client_id),
        )
    if cursor.rowcount == 0:
        logger.warning(
            "client_store: delete_client matched no row for advisor=%s client_id=%s "
            "(already deleted, or client_id belongs to a different advisor)",
            advisor_id,
            client_id,
        )
    else:
        logger.debug(
            "client_store: deleted client %s for advisor %s", client_id, advisor_id
        )
