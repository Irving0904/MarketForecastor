"""Google OAuth login.

Every advisor signs in with Google; `client_store.py` partitions all client
data by the resulting Google `sub` claim (the stable, never-reused unique
user id -- deliberately not email, which can change or be reassigned).

Route map (registered onto the FastAPI app in main.py, alongside the Gradio
app mounted at /app -- see that file for why /app rather than /):
  GET /             -- landing page: redirects into /app if already signed
                       in, otherwise a "Sign in with Google" link.
  GET /auth/login   -- starts the OAuth redirect to Google.
  GET /auth/callback -- Google redirects back here with an auth code.
  GET /auth/logout  -- clears the session, back to the landing page.

The session cookie holds only `sub` and `email` -- no tokens, nothing else.
`sub` is what every client_store call is keyed on; `email` is display-only
(the "logged in as" line in ui/app.py).
"""

import logging
import secrets

from authlib.integrations.starlette_client import OAuth
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from market_forecaster.config import get_google_oauth_credentials

logger = logging.getLogger(__name__)

_client_id, _client_secret = get_google_oauth_credentials()

oauth = OAuth()
oauth.register(
    name="google",
    client_id=_client_id,
    client_secret=_client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def get_current_advisor(request: Request) -> str | None:
    """The `auth_dependency` passed to gr.mount_gradio_app -- its return
    value becomes `request.username` inside every Gradio event handler
    that declares a `request: gr.Request` parameter. None means Gradio
    denies the request; in practice advisors reach /app only via the
    landing page's redirect, which already checked this same session key."""
    return request.session.get("sub")


async def landing_page(request: Request):
    if request.session.get("sub"):
        return RedirectResponse("/app")
    return HTMLResponse(
        "<html><body style='font-family: sans-serif; text-align: center; "
        "margin-top: 15vh;'>"
        "<h2>📈 Portfolio Advisor</h2>"
        "<p><a href='/auth/login'>Sign in with Google</a></p>"
        "</body></html>"
    )


async def login(request: Request):
    # OIDC nonce (replay protection for the ID token) is NOT handled
    # automatically by authlib's Starlette client -- unlike the CSRF
    # `state` parameter, which is. Stashing it in the session here and
    # verifying it in the callback via parse_id_token(..., nonce=...) is
    # on us to do explicitly.
    nonce = secrets.token_urlsafe(16)
    request.session["nonce"] = nonce
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri, nonce=nonce)


async def callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    nonce = request.session.pop("nonce", None)
    # parse_id_token (not just reading token["userinfo"]) is what actually
    # verifies the nonce claim against what we generated in login() above.
    userinfo = await oauth.google.parse_id_token(token, nonce=nonce)
    request.session["sub"] = userinfo["sub"]
    request.session["email"] = userinfo.get("email", "")
    logger.info("auth: signed in advisor sub=%s email=%s", userinfo["sub"], userinfo.get("email"))
    return RedirectResponse("/app")


async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")
