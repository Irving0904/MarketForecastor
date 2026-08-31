"""Formatting the Trace tab's HTML from the session's Q&A log."""

import html

ROUTE_LABELS = {
    "portfolio": "📊 Portfolio loaded",
    "straight": "⚡ Quick lookup · ReAct",
    "tot": "🌳 Strategy analysis · Tree-of-Thought",
    "prompt": "💬 Needs portfolio",
    "error": "⚠️ Error",
}

RESPONSE_PREVIEW_LENGTH = 200


def _format_response(response: str) -> str:
    """Long responses collapse behind a native <details> toggle — the
    preview itself is the clickable <summary>, so no separate 'Show more'
    label duplicates the text."""
    if len(response) <= RESPONSE_PREVIEW_LENGTH:
        return f"<div class='trace-response'>{response}</div>"
    preview = response[:RESPONSE_PREVIEW_LENGTH].rstrip() + "…"
    return (
        "<details class='trace-details'>"
        f"<summary class='trace-response-preview'>{preview}</summary>"
        f"<div class='trace-response'>{response}</div>"
        "</details>"
    )


def format_trace(trace_log: list[dict]) -> str:
    if not trace_log:
        return (
            "<div class='trace-empty'>No questions yet — the trace will "
            "appear here as you chat.</div>"
        )
    cards = []
    for entry in reversed(trace_log):
        badge = ROUTE_LABELS.get(entry["route"], entry["route"])
        question = html.escape(entry["question"])
        response = html.escape(entry["response"])
        cards.append(
            "<div class='trace-entry'>"
            f"<span class='trace-badge'>{badge}</span>"
            f"<div class='trace-question'>🧑 {question}</div>"
            f"{_format_response(response)}"
            "</div>"
        )
    return "".join(cards)
