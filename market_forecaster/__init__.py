"""Loads .env before any submodule of this package is imported.

Must happen here, not in config.py: several submodules (e.g. agents/router.py)
import langchain_core directly, before ever importing config.py. langsmith's
env-var lookup is @lru_cache'd, and langchain_core can trigger a tracing
check at import time — if that happens before .env is loaded, "tracing
disabled" gets cached for the life of the process no matter what
LANGSMITH_TRACING is set to afterward. __init__.py is the one place
Python guarantees runs before any submodule of this package does.
"""

from dotenv import load_dotenv

load_dotenv()
