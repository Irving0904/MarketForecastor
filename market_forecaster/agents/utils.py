"""Shared helpers for LangChain agent responses."""


def extract_text(content) -> str:
    """LangChain message `.content` is typed as a plain string OR a list of
    content blocks (e.g. `[{"type": "text", "text": "..."}]`) depending on
    how the model structures its reply. Normalize either shape to plain
    text so callers can always treat it as a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)
