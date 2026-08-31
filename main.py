"""Entry point: launches the Portfolio Advisor Gradio app."""

import logging

import gradio as gr

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
    demo.launch(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
