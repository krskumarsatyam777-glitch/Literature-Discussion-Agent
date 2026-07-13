"""
Structured output parser for generating reading diary entries.
"""

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class DiaryEntry(BaseModel):

    author: str = Field(
        description="Author or poet's name, or 'Unknown' if not identified"
    )
    title: str = Field(
        description="Title of the poem/work discussed, or 'Unknown' if not identified"
    )
    takeaway: str = Field(
        description="One-line summary of what was discussed or the key insight"
    )

parser = PydanticOutputParser(pydantic_object=DiaryEntry)

diary_prompt = ChatPromptTemplate.from_template(
    """Based on this conversation exchange, extract a structured diary entry.

User asked: {user_query}
Agent responded: {agent_response}

{format_instructions}"""
)