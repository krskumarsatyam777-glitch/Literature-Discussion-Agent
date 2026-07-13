"""
Builds the Literature Discussion Agent.
"""

import re
import threading

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from configure import (
    DEEPSEEK_API_KEY,
    PRIMARY_MODEL,
    DEEPSEEK_BASE_URL,
)

from prompts import SYSTEM_PROMPT
from parser import diary_prompt, parser

from utils import (
    log_chat,
    log_diary_entry,
)

from tools import (
    search_wikisource_tool,
    get_wikisource_page_content_tool,
    get_author_info_tool,
    extract_text_from_upload_tool,
    search_thematic_image_tool,
    search_web_tiered_tool,
    recommend_similar_works_tool,
)

# Language Model

llm = ChatOpenAI(
    model=PRIMARY_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0,
    max_retries=2,
).bind(parallel_tool_calls=True)

# Agent Tools

tools = [
    search_wikisource_tool,
    get_wikisource_page_content_tool,
    get_author_info_tool,
    extract_text_from_upload_tool,
    search_thematic_image_tool,
    search_web_tiered_tool,
    recommend_similar_works_tool,
]

# Reading Diary Chain

diary_chain = diary_prompt | llm | parser

# Agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


def extract_image_from_messages(result_messages):

    images = []

    for msg in result_messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        urls = re.findall(
            r"Image URL \d+:\s*(\S+)",
            content,
        )
        photographers = re.findall(
            r"Photographer \d+:\s*(.+)",
            content,
        )
        for url, photographer in zip(urls, photographers):
            images.append(
                {
                    "url": url,
                    "photographer": photographer.strip(),
                }
            )

    return images

def chat(query: str, history: list, image_path: str | None = None) -> tuple[str, str | None]:
    """
    Process a user query using the Literature Discussion Agent.

    Parameters
    ----------
    query : str
        Current user query.

    history : list
        Previous conversation history.

    Returns
    -------
    tuple
        (response, image_url)
    """

    log_chat("user", query)

    if image_path is not None:
        extracted_text = extract_text_from_upload_tool.invoke(
            {
                "image_path": image_path,
            }
        )

        query = (
            "The user uploaded an image of a literary work.\n\n"
            "Extracted Text:\n"
            f"{extracted_text}\n\n"
            "User Question:\n"
            f"{query}"
        )

    messages = history + [
        {
            "role": "user",
            "content": query,
        }
    ]

    payload = {
        "messages": messages,
    }

    try:
        result = agent.invoke(payload)

    except Exception as e:
        print(f"Agent invocation failed: {e}")

        error_response = (
            "I'm having trouble reaching my language model right now. "
            "Please try again in a moment."
        )

        log_chat("agent", error_response)

        return error_response, None

    image_data = extract_image_from_messages(
        result["messages"]
    )

    message = result["messages"][-1]

    response = (
        message.content
        if isinstance(message.content, str)
        else str(message.content)
    )

    log_chat("agent", response)

    # Diary logging runs in the background — don't make the user wait for it
    def _log_diary_background():
        try:
            log_diary_entry(query, response, diary_chain)
        except Exception as e:
            print(f"Diary logging failed: {e}")

    threading.Thread(target=_log_diary_background, daemon=True).start()

    return response, image_data

import time

def timed_llm_invoke(*args, **kwargs):
    start = time.time()
    result = llm.invoke(*args, **kwargs)
    print(f"[LLM CALL] took {time.time() - start:.2f}s")
    return result