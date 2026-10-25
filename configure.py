"""
Project Configuration

Loads environment variables and stores
global configuration settings used across
the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# LangSmith Tracing
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "literature-agent"
)


# Language Model
LLM_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

