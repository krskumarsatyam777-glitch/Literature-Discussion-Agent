"""
Streamlit interface for the Literature Discussion Agent.
"""

import re
import random
import tempfile
import streamlit as st

from agent import chat
from utils import (
    export_chat,
    export_diary,
)

st.set_page_config(
    page_title="Literature Discussion Agent",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Literature Discussion Agent")

st.markdown(
    """
Discuss novels, poems, authors and literary themes with
source-grounded responses.
"""
)

LITERARY_FACTS = [
    "🌙 Did you know? Dreams have inspired some of literature's greatest works."
    " Mary Shelley's *Frankenstein*, Samuel Taylor Coleridge's *Kubla Khan*, and"
    " Robert Louis Stevenson's *Strange Case of Dr Jekyll and Mr Hyde* all have"
    " famous connections to dreams.",

    "🎭 Did you know? The dramatic structure found in many of Shakespeare's"
    " plays continues to influence countless modern films, television series"
    ", and novels, proving that great storytelling techniques can endure for"
    " centuries.",

    "✍️ Did you know? Some of history's greatest writers became truly"
    " celebrated only after their deaths. Franz Kafka, Emily Dickinson,"
    " Fernando Pessoa, and John Keats all left works whose influence"
    " grew immensely after they were gone.",

    "🚫 Did you know? Many literary classics were once banned or censored."
    " *1984*, *Ulysses*, *Lady Chatterley's Lover*, and *Catch-22* all faced"
    " restrictions before becoming some of the world's most celebrated books.",

    "💭 Think about it... How many stories—in novels, poems, films,"
    " songs, conversations, and your own life experiences—have shaped"
    " the person you are today?",

    "🎨 Did you know? Throughout history, literature has often challenged"
    " social norms through symbolism. In Indian literary traditions—from Sufi"
    " poetry to Harivansh Rai Bachchan's *Madhushala* and many other works—wine"
    " is often a symbol of divine love, spiritual ecstasy, freedom, or life's"
    " journey rather than merely a drink.",

    "📖 A thought... One of art's greatest strengths is that the same work can"
    " hold different meanings for different people. Every revisit—and every stage"
    " of life—can reveal a new layer of understanding.",

    "❤️ Think about it... Love has remained one of literature's oldest and"
    " most enduring themes. Across centuries and cultures, writers continue to"
    " discover new ways to explore the same emotion."
]

if "messages" not in st.session_state:
    st.session_state.messages = []

def strip_markdown_images(text: str) -> str:
    """Remove markdown image syntax and bare image URLs from response text —
    the image is displayed separately via st.image(), so leaving these in
    causes the same image to render twice (once via markdown, once explicitly)."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # ![alt](url)
    text = re.sub(r"https?://images\.unsplash\.com/\S+", "", text)  # bare Unsplash URLs
    return text.strip()

st.markdown(
    """
<style>
.loading-card {
    border: 1px solid rgba(150,150,150,0.25);
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
    margin-bottom: 18px;
    background-color: rgba(250,250,250,0.03);
}
.loading-header {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 14px;
}
/* Book animation */
.loading-track {
    position: relative;
    width: 100%;
    height: 36px;
    margin: 8px 0 16px;
    overflow: hidden;
}
.bookshelf {
    position: absolute;
    right: 0;
    top: 2px;
    font-size: 28px;
}
.moving-book {
    position: absolute;
    left: -35px;
    top: 2px;
    font-size: 28px;
    animation: moveBook 8s linear infinite;
}
@keyframes moveBook {
    0% {
        left: -35px;
    }

    100% {
        left: calc(100% - 45px);
    }
}
.loading-fact {
    line-height: 1.6;
    font-size: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:

    st.header("Options")

    if st.button("Export Chat"):
        filename = export_chat()
        st.success(f"Saved as {filename}")

    if st.button("Export Reading Diary"):
        filename = export_diary()
        st.success(f"Saved as {filename}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("image_data"):
            cols = st.columns(len(message["image_data"]))
            for col, image in zip(cols, message["image_data"]):
                with col:
                    st.image(
                        image["url"],
                        width="stretch",
                    )
                    st.caption(
                        f"Photo by {image['photographer']} • Unsplash"
                    )

chat_input = st.chat_input(
    "Ask about a poem, novel, author... (attach an image to identify a poem)",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
)

if chat_input:
    prompt = chat_input.text
    image_path = None

    if chat_input.files:
        uploaded_file = chat_input.files[0]
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as temp_file:
            temp_file.write(uploaded_file.read())
            image_path = temp_file.name

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        loading_placeholder = st.empty()

        loading_placeholder.markdown(
            f"""
            <div class="loading-card">
               <div class="loading-header">
                    📚 While you wait...
                </div>
                <div class="loading-track">
                    <span class="moving-book">📖</span>
                    <span class="bookshelf">📚</span>
                </div>
                <div class="loading-fact">
                    {random.choice(LITERARY_FACTS)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        history = [
            {
                "role": m["role"],
                "content": m["content"],
            }
            for m in st.session_state.messages[:-1]
        ]

        response, image_data = chat(
            prompt,
            history,
            image_path,
        )

        loading_placeholder.empty()
        clean_response = strip_markdown_images(response)
        st.markdown(clean_response)

        if image_data:
            cols = st.columns(len(image_data))
            for col, image in zip(cols, image_data):
                with col:
                    st.image(
                        image["url"],
                        width="stretch",
                    )
                    st.caption(
                        f"Photo by {image['photographer']} • Unsplash"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": clean_response,
            "image_data": image_data,
        }
    )

    st.rerun()