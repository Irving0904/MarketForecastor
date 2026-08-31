"""SQLite persistence for client sessions — long-term memory across app
restarts. `clients_state` in ui/app.py stays the in-memory source of truth
during a session; this module is what makes it durable.

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
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                chat_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
    logger.info("client_store: initialized database at %s", DB_PATH)


def save_client(client_id: str, client: dict) -> None:
    """Upsert one client's full session state."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO clients (id, name, profile_json, trace_json, chat_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                profile_json=excluded.profile_json,
                trace_json=excluded.trace_json,
                chat_json=excluded.chat_json,
                updated_at=excluded.updated_at
            """,
            (
                client_id,
                client.get("name", ""),
                json.dumps(client.get("profile", {}), default=str),
                json.dumps(client.get("trace", []), default=str),
                json.dumps(client.get("chat", []), default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    logger.debug("client_store: saved client %s (%s)", client_id, client.get("name"))


def load_all_clients() -> dict[str, dict]:
    """Read every persisted client into the shape clients_state expects."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, profile_json, trace_json, chat_json FROM clients"
        ).fetchall()
    clients = {}
    for client_id, name, profile_json, trace_json, chat_json in rows:
        clients[client_id] = {
            "name": name,
            "profile": json.loads(profile_json),
            "trace": json.loads(trace_json),
            "chat": json.loads(chat_json),
        }
    logger.info("client_store: loaded %d client(s) from disk", len(clients))
    return clients


def search_clients(query: str) -> list[tuple[str, str]]:
    """(name, id) pairs for clients whose name contains query, case-insensitive."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, id FROM clients WHERE name LIKE ? ORDER BY name",
            (f"%{query}%",),
        ).fetchall()
    return [(name, client_id) for name, client_id in rows]


def delete_client(client_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM clients WHERE id = ?", (client_id,))
    logger.debug("client_store: deleted client %s", client_id)
