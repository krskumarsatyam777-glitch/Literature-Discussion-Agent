"""
Utility functions for chat logging and reading diary management.
"""
from parser import parser
from datetime import datetime

# Chat History
chat_history = []

def log_chat(role: str, content: str):
    """
       Store a chat message with timestamp.
       """
    chat_history.append(
        {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    )

def export_chat() -> str:
    """
        Build chat history as Markdown content, ready for download.
        """
    lines = ["#Chat Export\n"]

    for entry in chat_history:
        role_label = "**You**" if entry["role"] == "user" else "**Agent**"
        lines.append(f"### {role_label} ({entry['timestamp']})\n{entry['content']}\n")

    return "\n".join(lines)

#reading diary
diary_entries = []

def log_diary_entry(user_query: str, agent_response: str, diary_chain):
    """
    Extract and store a structured reading diary entry from a conversation.
    """

    try:
        entry = diary_chain.invoke(
            {
                "user_query": user_query,
                "agent_response": agent_response,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        if (
            entry.author.lower() != "unknown"
            or entry.title.lower() != "unknown"
        ):
            diary_entries.append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "author": entry.author,
                    "title": entry.title,
                    "takeaway": entry.takeaway,
                }
            )

    except Exception as e:
        print(f"Diary extraction skipped (non-critical): {e}")

def export_diary() -> str:
    """
       Build reading diary as Markdown table content, ready for download.
       """
    lines = [
        "# Reading Diary\n",
        "| Date | Author | Title | Takeaway |",
        "|------|--------|-------|-----------|"
    ]

    for e in diary_entries:
        lines.append(
            f"| "
            f"{e['date']} "
            f"| {e['author']} "
            f"| {e['title']} "
            f"| {e['takeaway']} |"
        )

    return "\n".join(lines)