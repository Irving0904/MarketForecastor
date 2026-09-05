"""Entry point: launches the Portfolio Advisor Gradio app behind Google
sign-in.

Gradio is mounted at /app, not / -- gr.mount_gradio_app's auth_dependency
is the documented mechanism for custom (non-username/password) auth, but
what it shows an unauthenticated visitor at the mount path isn't something
this project can verify without a live Google login. Owning / as a plain
FastAPI route sidesteps that entirely: it's a landing page we control and
can test, that redirects into /app once signed in.
"""

import logging

import fastapi
import gradio as gr
import uvicorn
from starlette.middleware.sessions import SessionMiddleware

from market_forecaster import auth
from market_forecaster.config import get_session_secret
from market_forecaster.guardrails import startup_checks
from market_forecaster.ui.app import create_app
from market_forecaster.ui.styles import CUSTOM_CSS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    # Fails loudly before the server ever accepts input if any agent is
    # wired to a write/execute-capable tool -- this app must stay read-only.
    startup_checks()
    demo = create_app()

    app = fastapi.FastAPI()
    # Must be added before mount_gradio_app / the /auth routes -- both
    # depend on request.session existing.
    app.add_middleware(
        SessionMiddleware,
        secret_key=get_session_secret(),
        same_site="lax",
        # Localhost-only setting: a Secure cookie is never sent back by
        # the browser over plain http, which would silently break the
        # entire login flow. Flip to True before any non-localhost deploy.
        https_only=False,
    )
    app.add_api_route("/", auth.landing_page, methods=["GET"])
    app.add_api_route("/auth/login", auth.login, methods=["GET"])
    app.add_api_route(
        "/auth/callback", auth.callback, methods=["GET"], name="auth_callback"
    )
    app.add_api_route("/auth/logout", auth.logout, methods=["GET"])

    gr.mount_gradio_app(
        app,
        demo,
        path="/app",
        auth_dependency=auth.get_current_advisor,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=CUSTOM_CSS,
    )
    uvicorn.run(app, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
