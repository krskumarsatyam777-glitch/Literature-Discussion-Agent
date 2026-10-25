# Literature Discussion Agent

**Live demo:** [Open the Literature Discussion Agent](https://literature-discussion-agent.onrender.com/)

> **Demo note:** The hosted service may take a little longer to respond after a period of inactivity. The demo's saved chats are shared between visitors, so please do not enter private or sensitive information.

A source-grounded literature discussion app. Ask about a poem, novel, play, author, or literary theme—or upload a photo of a poem. The agent retrieves literary text and background information from online sources before discussing the work conversationally.

Built with **LangGraph**, **DeepSeek**, **Streamlit**, and **PostgreSQL**.

## Features

- **Source-grounded answers:** Uses retrieved material from sources such as Wikisource, Wikipedia, Rekhta, Hindwi, Literary Hub, and web search instead of relying on the model's memory alone.
- **Smart routing:** Classifies messages as new topics, related works, or follow-ups. Follow-ups that the current conversation can answer may skip retrieval.
- **Literature scope guard:** Politely declines requests outside the app's literary focus.
- **Parallel retrieval:** Searches Wikisource and Wikipedia, and searches trusted Hindi and Urdu literary sources where relevant.
- **Sufficiency checks and web fallback:** Uses open-web search when retrieved sources are insufficient. For full-text requests, it avoids substituting commentary when the complete text cannot be verified.
- **Image upload and OCR:** Accepts poem images in English or Hindi and uses Tesseract OCR to help identify the work. OCR requires the Tesseract system executable and language data to be installed in the runtime.
- **Related images with credits:** Retrieves images from Wikimedia Commons and Unsplash with attribution and licence information where available.
- **Poem formatting:** Preserves poem line breaks in a styled quote block.
- **Saved chats:** Stores conversation history in PostgreSQL and provides a chat sidebar and delete option.
- **Live progress:** Displays status updates and streams the answer as it is generated.
- **Tracing and feedback:** Can send traces to LangSmith when tracing is configured; answers also include thumbs-up/thumbs-down feedback.

## How it works

![Agent graph flowchart](screenshots/workflow.png)

The app uses a LangGraph workflow with LLM-based decisions for analysis, routing, candidate selection, and retrieval sufficiency. Conversation state is checkpointed in PostgreSQL.

## Tech stack

| Area | Tools |
|---|---|
| Orchestration | LangGraph, LangChain Core |
| Language model | DeepSeek via its OpenAI-compatible API |
| Interface | Streamlit |
| Storage | PostgreSQL for LangGraph checkpoints and the chat list |
| Retrieval | Wikisource, Wikipedia, Rekhta, Hindwi, Literary Hub, Tavily web search |
| Images | Wikimedia Commons, Unsplash |
| OCR | Tesseract (English and Hindi) |
| Tracing | LangSmith (optional) |

## Project structure

```text
app.py          Streamlit interface, chat sidebar, and streaming
graph.py        LangGraph workflow: analysis, routing, retrieval, and generation
tools.py        Retrieval, OCR, and image-search tools
prompts.py      System and analyzer prompts
thread.py       Chat list stored in PostgreSQL
configure.py    Model settings
.streamlit/     Streamlit theme configuration
screenshots/    Workflow diagram and project screenshots
```

## Local setup

### Requirements

- Python 3.11 or newer
- A PostgreSQL database reachable from your computer
- API keys for the services you use
- Tesseract OCR with English and Hindi language data if you want to use image upload

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/krskumarsatyam777-glitch/Literature-Discussion-Agent.git
cd Literature-Discussion-Agent
python -m venv venv
```

Activate the environment:

```bash
# macOS / Linux
source venv/bin/activate
```

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

### 2. Install Tesseract OCR (optional, for image uploads)

On Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-hin
```

On Windows, install Tesseract OCR with the English and Hindi language data. If `tesseract.exe` is not on your `PATH`, set `TESSERACT_CMD` in `.env` to its path, for example:

```env
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

`pytesseract` is a Python wrapper; it does not install the Tesseract executable itself. A hosted deployment must also have Tesseract and the required language data installed for OCR to work there.

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in the values. Do not commit `.env` or share its secrets.

```env
# Required for the core app
DEEPSEEK_API_KEY=your_deepseek_api_key
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
TAVILY_API_KEY=your_tavily_api_key

# Needed for related-image search
UNSPLASH_ACCESS_KEY=your_unsplash_access_key

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=literature-agent-v2

# Optional: set only if Tesseract is not on PATH
TESSERACT_CMD=
```

Use the complete connection string supplied by your PostgreSQL provider. Keep any connection parameters the provider includes, such as SSL settings. The example above is a placeholder, not a working database URL. For a hosted deployment, set these values in the host's environment-variable settings rather than relying on your local `.env` file.

### 4. Run the app

```bash
streamlit run app.py
```

The app initializes its database tables on startup. The PostgreSQL database must be reachable and the `DATABASE_URL` must be valid.

## Deployment on Render

The live demo is hosted at [literature-discussion-agent.onrender.com](https://literature-discussion-agent.onrender.com/).

Set the required environment variables in **Render → your service → Environment**. At minimum, configure `DATABASE_URL`, `DEEPSEEK_API_KEY`, and `TAVILY_API_KEY`; add `UNSPLASH_ACCESS_KEY` for related-image search and the `LANGSMITH_*` variables if you want tracing. The local `.env` file is not automatically uploaded to Render.

A typical Streamlit start command on Render is:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

For image OCR on Render, ensure the deployed runtime installs Tesseract and the English/Hindi language data. Installing `pytesseract` through `requirements.txt` alone is not sufficient.

## Example queries

- Explain Ghalib's ghazal *Hazaron Khwahishein Aisi*.
- Discuss the ending of *Hamlet*.
- Compare *Hamlet* and *Macbeth*.
- Give me the full text of *Ozymandias* and discuss it.
- What do critics say about *The Waste Land*?
- Upload a photo of a poem and ask what it is.

## Current limitations

- **Heuristic retrieval:** Title matching, page-type detection, and thresholds rely on hand-set rules and have not been tuned on a labelled dataset.
- **Wikipedia matching:** The selected article may occasionally be wrong for ambiguous titles or names.
- **Long works:** Plays and novels split across multiple Wikisource subpages are not yet followed page by page in every case.
- **Source coverage:** Trusted literary sites such as Rekhta, Hindwi, and Literary Hub are searched only for selected language/query types; English queries rely more heavily on Wikisource, Wikipedia, and web fallback.
- **Latency:** Some non-English queries may take longer because language-specific sources are searched sequentially.
- **OCR:** Supports English and Hindi only; accuracy depends on image quality and on Tesseract being installed in the runtime.
- **Shared chat history:** The public demo uses a shared chat list rather than per-user histories. Do not enter private or sensitive content.
- **Evaluation:** Answer quality has been checked manually; an automated evaluation suite has not yet been added.

## Future improvements

### Retrieval quality

- Replace heuristic matching with semantic ranking using multilingual embeddings and, optionally, a cross-encoder reranker.
- Choose thresholds using a labelled set of queries and correct pages.
- Verify selected Wikipedia articles against the intended work or author.
- Follow index pages and subpages for multi-page works, including acts and chapters.

### Coverage

- Add more trusted literary sources and configure sources by language, for example Project Gutenberg, Poetry Foundation, and Kavita Kosh.
- Expand supported languages, including OCR languages.

### Speed

- Search language editions in parallel and cache repeated lookups.

### Evaluation

- Create an offline evaluation set of representative queries.
- Add online evaluators for hallucination, over-refusal, and latency.

### Interface

- Generate chat titles from the work and author rather than the first message.
- Add per-user chat history.

## Acknowledgements

LangGraph, LangChain, DeepSeek, Streamlit, Wikisource, Wikipedia, Wikimedia Commons, Rekhta, Hindwi, Literary Hub, Tavily, Unsplash, and Tesseract OCR.
