"""Gradio Blocks layout: chat on the left, Clients + Trace tabs on the right.

Multi-client model: `clients_state` is the single source of truth — a dict
of `{client_id: {"name", "profile", "trace", "chat"}}`. `active_client_id`
names which entry is currently loaded into the visible chat/profile/trace.

Switching, creating, or deleting a client writes the target conversation
into `chat_interface.chatbot_value` rather than the visible `chatbot`
component directly. `ChatInterface` keeps its own hidden `gr.State`
(`chatbot_state`) as the actual history it feeds to `fn` — writing only to
the visible `chatbot` component updates the display but leaves that hidden
state stale, so the previous client's messages resurface on the next send.
`chatbot_value` is the hook `ChatInterface` itself documents for exactly
this ("update the chatbot value from other events outside of
gr.ChatInterface") — it propagates to both the display and the hidden
history state internally. The chat textbox is gated the same way, via
`chat_interface.textbox`, so a message literally cannot be sent until a
client is selected — not just rejected after the fact.
"""

import gradio as gr

from market_forecaster.core import client_store
from market_forecaster.core.clients import (
    empty_client,
    name_exists,
    new_client_id,
    radio_choices,
)
from market_forecaster.core.orchestrator import respond
from market_forecaster.core.trace import format_trace
from market_forecaster.data import vector_store
from market_forecaster.ui.styles import HEADER_HTML

NO_CLIENT_PLACEHOLDER = "Select or create a client to begin chatting"
CHAT_PLACEHOLDER = "Type a message..."


def chat_fn(
    message,
    history,
    profile_state,
    trace_state,
    clients_state,
    active_client_id,
    progress=gr.Progress(),
):
    if not active_client_id:
        return (
            "Create or select a client from the 👥 Clients tab before "
            "chatting.",
            profile_state,
            trace_state,
            format_trace(trace_state or []),
            clients_state,
        )

    reply, updated_profile_state, route = respond(
        message, history, profile_state, active_client_id, progress=progress
    )
    updated_trace_state = (trace_state or []) + [
        {"question": message, "response": reply, "route": route}
    ]
    updated_chat = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]

    clients_state = dict(clients_state or {})
    if active_client_id in clients_state:
        clients_state[active_client_id] = {
            **clients_state[active_client_id],
            "profile": updated_profile_state,
            "trace": updated_trace_state,
            "chat": updated_chat,
        }
        client_store.save_client(active_client_id, clients_state[active_client_id])

        # Embed only this turn — never the whole history — into a
        # per-client-growing id space, so this is always a pure append.
        turn_index = len(history) // 2
        vector_store.add_chunks(
            "client_history",
            ids=[
                f"{active_client_id}_{turn_index}_user",
                f"{active_client_id}_{turn_index}_assistant",
            ],
            documents=[message, reply],
            metadatas=[
                {"client_id": active_client_id, "role": "user", "turn": turn_index},
                {
                    "client_id": active_client_id,
                    "role": "assistant",
                    "turn": turn_index,
                },
            ],
        )

    return (
        reply,
        updated_profile_state,
        updated_trace_state,
        format_trace(updated_trace_state),
        clients_state,
    )


def refresh_clients_on_load():
    """`clients_state`/`client_selector` are seeded once from disk at
    `create_app()` time (server process startup), and Gradio hands every
    new browser session a copy of that same static default -- a client
    created in one session (or by a prior run of this same session before
    a reload) never shows up for a session that starts afterward, since
    nothing re-reads the DB after startup. Re-reading it here, on every
    `demo.load`, keeps a fresh session in sync with what's actually on
    disk instead of a stale process-startup snapshot."""
    fresh_clients = client_store.load_all_clients()
    return fresh_clients, gr.update(choices=radio_choices(fresh_clients))


def create_client(name, clients_state):
    clients_state = dict(clients_state or {})
    name = (name or "").strip() or f"Client {len(clients_state) + 1}"

    if name_exists(name, clients_state):
        return (
            gr.skip(),  # clients_state
            gr.skip(),  # active_client_id
            gr.skip(),  # profile_state
            gr.skip(),  # trace_state
            gr.skip(),  # trace_html
            gr.skip(),  # chatbot_value
            gr.skip(),  # client_selector
            gr.skip(),  # chat_textbox
            f"⚠️ A client named '{name}' already exists — pick a different name.",
            gr.skip(),  # new_client_name (leave the typed text as-is)
        )

    client_id = new_client_id()
    clients_state[client_id] = empty_client(name)
    client_store.save_client(client_id, clients_state[client_id])
    return (
        clients_state,
        client_id,
        {},
        [],
        format_trace([]),
        [],
        gr.update(choices=radio_choices(clients_state), value=client_id),
        gr.update(interactive=True, placeholder=CHAT_PLACEHOLDER),
        "",
        "",
    )


def switch_client(selected_id, clients_state):
    clients_state = clients_state or {}
    client = clients_state.get(selected_id, empty_client(""))
    trace = client.get("trace", [])
    return (
        selected_id,
        client.get("profile", {}),
        trace,
        format_trace(trace),
        client.get("chat", []),
        gr.update(interactive=True, placeholder=CHAT_PLACEHOLDER),
        "",
    )


def delete_client(active_client_id, clients_state):
    clients_state = dict(clients_state or {})
    clients_state.pop(active_client_id, None)
    client_store.delete_client(active_client_id)
    remaining = list(clients_state.items())
    if remaining:
        new_id, new_client = remaining[0]
        profile, trace, chat = (
            new_client.get("profile", {}),
            new_client.get("trace", []),
            new_client.get("chat", []),
        )
        textbox_update = gr.update(interactive=True, placeholder=CHAT_PLACEHOLDER)
    else:
        new_id, profile, trace, chat = None, {}, [], []
        textbox_update = gr.update(interactive=False, placeholder=NO_CLIENT_PLACEHOLDER)
    return (
        clients_state,
        new_id,
        profile,
        trace,
        format_trace(trace),
        chat,
        gr.update(choices=radio_choices(clients_state), value=new_id),
        textbox_update,
        "",
    )


def filter_clients(query: str, clients_state: dict):
    """Live-filters the client-switcher list by name, in-memory — the
    already-loaded clients_state, not a fresh DB query per keystroke."""
    query = (query or "").strip().lower()
    clients_state = clients_state or {}
    if not query:
        return gr.update(choices=radio_choices(clients_state))
    matches = {
        cid: c for cid, c in clients_state.items() if query in c["name"].lower()
    }
    return gr.update(choices=radio_choices(matches))


def create_app() -> gr.Blocks:
    client_store.init_db()
    loaded_clients = client_store.load_all_clients()

    with gr.Blocks(title="Portfolio Advisor", fill_height=True) as demo:
        gr.HTML(HEADER_HTML)
        profile_state = gr.State({})
        trace_state = gr.State([])
        clients_state = gr.State(loaded_clients)
        active_client_id = gr.State(None)
        trace_html = gr.HTML(
            value=format_trace([]), elem_id="trace-panel", render=False
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=560, label="Chat")
                chat_interface = gr.ChatInterface(
                    fn=chat_fn,
                    additional_inputs=[
                        profile_state,
                        trace_state,
                        clients_state,
                        active_client_id,
                    ],
                    additional_outputs=[
                        profile_state,
                        trace_state,
                        trace_html,
                        clients_state,
                    ],
                    chatbot=chatbot,
                )
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("👥 Clients"):
                        new_client_name = gr.Textbox(
                            label="New client",
                            placeholder="Client name (e.g. Jane Doe)",
                        )
                        new_client_btn = gr.Button("+ New Client", size="sm")
                        client_status = gr.Markdown("")
                        client_search = gr.Textbox(
                            label="Search clients",
                            placeholder="Type a client name…",
                        )
                        client_selector = gr.Radio(
                            choices=radio_choices(loaded_clients),
                            label="Switch client",
                            value=None,
                        )
                        delete_client_btn = gr.Button(
                            "🗑️ Delete Selected Client",
                            variant="stop",
                            size="sm",
                        )
                    with gr.Tab("🔍 Trace"):
                        trace_html.render()

        chatbot_value = chat_interface.chatbot_value
        chat_textbox = chat_interface.textbox

        # No client is active on a fresh page load — block chat until one
        # is created or selected, rather than accepting and nudging.
        demo.load(
            fn=lambda: gr.update(
                interactive=False, placeholder=NO_CLIENT_PLACEHOLDER
            ),
            outputs=[chat_textbox],
        )
        demo.load(
            fn=refresh_clients_on_load,
            outputs=[clients_state, client_selector],
        )

        new_client_btn.click(
            fn=create_client,
            inputs=[new_client_name, clients_state],
            outputs=[
                clients_state,
                active_client_id,
                profile_state,
                trace_state,
                trace_html,
                chatbot_value,
                client_selector,
                chat_textbox,
                client_status,
                new_client_name,
            ],
        )

        client_search.change(
            fn=filter_clients,
            inputs=[client_search, clients_state],
            outputs=[client_selector],
        )

        client_selector.select(
            fn=switch_client,
            inputs=[client_selector, clients_state],
            outputs=[
                active_client_id,
                profile_state,
                trace_state,
                trace_html,
                chatbot_value,
                chat_textbox,
                client_status,
            ],
        )

        delete_client_btn.click(
            fn=delete_client,
            inputs=[active_client_id, clients_state],
            outputs=[
                clients_state,
                active_client_id,
                profile_state,
                trace_state,
                trace_html,
                chatbot_value,
                client_selector,
                chat_textbox,
                client_status,
            ],
        )

    return demo
