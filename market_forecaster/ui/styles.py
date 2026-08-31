"""CSS and static HTML fragments for the Gradio UI."""

CUSTOM_CSS = """
#app-header { text-align: center; padding: 4px 0 12px 0; }
#app-header h1 { margin-bottom: 4px; }
#app-header p { color: var(--body-text-color-subdued); margin-top: 0; }
#trace-panel {
    max-height: 640px;
    overflow-y: auto;
    padding: 4px 10px;
}
.trace-entry {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 12px;
    background: var(--background-fill-secondary);
}
.trace-badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    margin-bottom: 8px;
}
.trace-question { font-weight: 600; margin-bottom: 6px; }
.trace-response {
    white-space: pre-wrap;
    color: var(--body-text-color-subdued);
    font-size: 14px;
    line-height: 1.4;
}
.trace-response-preview {
    white-space: pre-wrap;
    color: var(--body-text-color-subdued);
    font-size: 14px;
    line-height: 1.4;
    cursor: pointer;
    list-style: none;
}
.trace-response-preview::-webkit-details-marker { display: none; }
.trace-response-preview::after {
    content: " ▸ show more";
    color: var(--body-text-color-subdued);
    font-size: 12px;
    font-style: italic;
}
.trace-details[open] .trace-response-preview::after {
    content: " ▾ show less";
}
.trace-details[open] .trace-response-preview {
    margin-bottom: 6px;
}
.trace-empty {
    text-align: center;
    color: var(--body-text-color-subdued);
    padding: 48px 12px;
}

/* Gradio's gr.Progress() overlay: these inner elements only exist in the
   DOM while a request is actively processing, so styling them can't get
   stuck visible after the fact. Deliberately not touching the outer
   .wrap.translucent dimming layer (its opacity is animated by Gradio
   itself) — instead this gives the progress content its own solid,
   high-contrast card so it reads clearly regardless of what's rendered
   behind it, which was the actual legibility problem. */
.progress-level {
    position: relative;
    z-index: 50;
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
}
.progress-text {
    color: var(--body-text-color) !important;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 10px;
    text-shadow: none !important;
}
.progress-level-inner {
    color: var(--body-text-color) !important;
}
.progress-bar-wrap {
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    border: 1px solid var(--border-color-primary);
}
.progress-bar {
    height: 100% !important;
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
}
"""

HEADER_HTML = """
<div id="app-header">
<h1>📈 Portfolio Advisor</h1>
<p>Paste a portfolio (CSV or ticker list) to get started, then ask follow-up questions about it.</p>
</div>
"""
