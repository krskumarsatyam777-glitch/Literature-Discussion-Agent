"""
Streamlit interface for the Literature Discussion Agent (LangGraph).
"""

import os
import tempfile
import uuid

from langsmith import Client

import requests
import streamlit as st

from graph import graph
from thread import delete_thread, init_threads_table, list_threads, upsert_thread

st.set_page_config(
    page_title="Literature Discussion Agent",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,400&display=swap');

    :root {
        --poem-bg: #D6CAB4;
        --poem-bar: #9A3F15;
    }

    .stApp {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.12'/%3E%3C/svg%3E");
        background-size: 140px 140px;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: transparent;
    }

    [data-testid="stChatMessage"] blockquote {
        background: var(--poem-bg);
        border-left: 3px solid var(--poem-bar);
        border-radius: 0;
        padding: 0.8rem 1.2rem;
        margin: 0.9rem 0;
        font-family: "Playfair Display", Georgia, serif;
        font-style: italic;
        font-synthesis: none;
        font-size: 1.05rem;
        line-height: 1.85;
    }
    [data-testid="stChatMessage"] blockquote p {
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

UTM = "utm_source=literature_discussion_agent&utm_medium=referral"
client = Client()

# Status lines shown while the graph runs (a node's line appears when it finishes).
STATUS = {
    "ocr": "Reading your image",
    "analyze": "Understanding your question",
    "wikisource": "Searching for sources",
    "wikipedia": "Searching for sources",
    "literary_sites": "Searching for sources",
    "web_opinions": "Searching the web",
    "web_fallback": "Searching the web",
    "evaluate": ["Interpreting what I found", "Thinking"],
}
FIXED_REPLY_NODES = {"refuse", "ocr_failed", "analysis_error"}


@st.cache_resource
def _init_db():
    init_threads_table()


_init_db()


# ======================================= Chat Management Helpers=========================================

def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def load_thread(thread_id: str):
    """Restore a saved conversation. Images are not stored, so only text returns."""
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    saved = (snapshot.values or {}).get("messages", [])
    st.session_state.thread_id = thread_id
    st.session_state.messages = [
        {
            "role": "user" if m.type == "human" else "assistant",
            "content": m.content,
            "images": [],
        }
        for m in saved
        if m.type in ("human", "ai")
    ]

def remove_thread(thread_id: str):
    """Delete a saved chat; if it is the open one, start a fresh chat."""
    delete_thread(thread_id)
    if st.session_state.thread_id == thread_id:
        new_chat()


@st.dialog("Delete this chat?")
def confirm_delete(thread_id: str, title: str):
    st.write(f"**{title}** will be permanently deleted.")
    yes_col, no_col = st.columns(2)
    if yes_col.button("Delete", type="primary", width="stretch"):
        remove_thread(thread_id)
        st.rerun()
    if no_col.button("Cancel", width="stretch"):
        st.rerun()

if "thread_id" not in st.session_state:
    new_chat()


# ========================================= Images ============================================

def track_unsplash_download(image: dict):
    """Unsplash asks apps to ping download_location when a photo is shown."""
    url = image.get("download_location")
    key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not url or not key:
        return
    try:
        requests.get(url, headers={"Authorization": f"Client-ID {key}"}, timeout=5)
    except Exception:
        pass


def render_images(images: list):
    if not images:
        return
    cols = st.columns(len(images))
    for col, image in zip(cols, images):
        with col:
            st.image(image["url"], width="stretch")
            if image["source"] == "unsplash":
                profile = f"{image['credit_url']}?{UTM}" if image.get("credit_url") else ""
                name = (
                    f"[{image['credit_name']}]({profile})"
                    if profile
                    else image["credit_name"]
                )
                st.caption(f"Photo by {name} on [Unsplash](https://unsplash.com/?{UTM})")
            else:
                artist = (image.get("credit_name") or "Unknown")[:120]
                license_text = image.get("license") or "License not specified"
                license_link = (
                    f"[{license_text}]({image['license_url']})"
                    if image.get("license_url")
                    else license_text
                )
                st.caption(
                    f"{artist} · {license_link} · "
                    f"[Wikimedia Commons]({image.get('page_url', '')})"
                )


# ====================================Streaming Responses and Feedback=====================================

RETRIEVAL_NODES = {
    "wikisource", "wikipedia", "literary_sites",
    "web_opinions", "web_fallback", "fetch_images",
}


def run_turn(prompt: str, image_path, status, answer_box):
    run_id = uuid.uuid4()
    thread_id = st.session_state.thread_id
    config = {
        "configurable": {"thread_id": thread_id},
        "run_id": run_id,
        "tags": ["lit-agent"],
        "metadata": {"thread_id": thread_id, "has_image": bool(image_path)},
    }
    inputs = {
        "messages": [("user", prompt or "Tell me about this text.")],
        "image_path": image_path,
    }

    shown = set()
    nodes_run = set()

    def show(label):
        if label not in shown:
            shown.add(label)
            status.write(f"✓ {label}")

    streamed = ""
    for mode, chunk in graph.stream(
        inputs, config, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node in chunk:
                nodes_run.add(node)
                labels = STATUS.get(node, [])
                for label in ([labels] if isinstance(labels, str) else labels):
                    show(label)
        else:  # "messages": tokens from the LLM
            token, meta = chunk
            # Only the answer node; the analyzer's output is not for the user.
            if meta.get("langgraph_node") == "generate" and token.content:
                # Backup in case the evaluate step sent no update.
                show("Interpreting what I found")
                show("Thinking")
                streamed += token.content
                answer_box.markdown(streamed + "▌")

    values = graph.get_state(config).values
    # The saved final message is the source of truth (covers the fixed replies,
    # which are not streamed as tokens).
    final = values["messages"][-1].content
    answer_box.markdown(final)

    # LangSmith: tag the run with route, tools used and whether the fallback fired.
    try:
        route = values.get("route") or "none"
        fallback = "web_fallback" in nodes_run
        tools = sorted(nodes_run & RETRIEVAL_NODES)
        client.update_run(
            run_id,
            tags=[f"route:{route}", "fallback" if fallback else "no-fallback"],
            extra={"metadata": {"tools_used": tools, "fallback_fired": fallback}},
        )
    except Exception:
        pass  # tracing problems must never break the app

    return final, values.get("images") or [], str(run_id)

def send_feedback(run_id: str, key: str):
    score = st.session_state.get(key)  # 1 = thumbs up, 0 = thumbs down
    if score is None:
        return
    try:
        client.create_feedback(run_id=run_id, key="user_score", score=score)
    except Exception:
        pass

#============================Page Layout ==============================================================

st.title("📚 Literature Discussion Agent")
st.markdown(
    """
Discuss novels, poems, authors and literary themes with
source-grounded responses.
"""
)

with st.sidebar:
    st.header("Chats")
    st.button("➕ New chat", on_click=new_chat, width="stretch")
    for t in list_threads():
        current = "▶ " if t["id"] == st.session_state.thread_id else ""
        title_col, delete_col = st.columns([5, 1])
        title_col.button(
            f"{current}{t['title']}",
            key=f"thread_{t['id']}",
            on_click=load_thread,
            args=(t["id"],),
            width="stretch",
        )
        if delete_col.button("🗑", key=f"delete_{t['id']}", help="Delete this chat"):
            confirm_delete(t["id"], t["title"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_images(message.get("images"))
        if message["role"] == "assistant" and message.get("run_id"):
            fb_key = f"fb_{message['run_id']}"
            st.feedback(
                "thumbs",
                key=fb_key,
                on_change=send_feedback,
                args=(message["run_id"], fb_key),
            )

chat_input = st.chat_input(
    "Ask about a poem, novel, author... (attach an image to identify a poem)",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
)

if chat_input:
    prompt = chat_input.text or ""
    image_path = None

    if chat_input.files:
        uploaded_file = chat_input.files[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(uploaded_file.read())
            image_path = temp_file.name

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "images": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt or "📷 Uploaded image")

    with st.chat_message("assistant"):
        status = st.status("Working on it…", expanded=True)
        answer_box = st.empty()
        try:
            answer, images, run_id = run_turn(prompt, image_path, status, answer_box)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"Sorry, that failed: {e}")
            st.stop()

        render_images(images)
        for image in images:
            if image["source"] == "unsplash":
                track_unsplash_download(image)

    upsert_thread(st.session_state.thread_id, (prompt or "Uploaded image")[:50])
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "images": images, "run_id": run_id}
    )
    st.rerun()