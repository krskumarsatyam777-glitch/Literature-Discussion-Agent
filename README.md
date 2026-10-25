# Literature Discussion Agent

A source-grounded literature discussion app. Ask about a poem, novel, play, author or literary theme (or upload a photo of a poem) and the agent retrieves the text and background from literary sources first, then discusses it in a conversational way.

Built with **LangGraph**, **DeepSeek**, **Streamlit** and **PostgreSQL**.

<!-- Add 1-2 screenshots from /screenshots here, e.g. ![Chat view](screenshots/your-file.png) -->

---

## Features

- **Source-grounded answers.** Responses are written from retrieved sources (Wikisource, Wikipedia, Rekhta, Hindwi, Literary Hub, web search) rather than from the model's memory alone.
- **Smart routing.** Each message is analysed and treated as a new topic, a related work, or a follow-up. Follow-ups that the conversation can already answer skip retrieval.
- **Scope guard.** Non-literary requests are politely declined.
- **Parallel retrieval.** Wikisource, Wikipedia and (for Hindi and Urdu) trusted literary sites are searched at the same time.
- **Sufficiency check and web fallback.** If the retrieved sources do not answer the request, the agent falls back to open-web search. For a full-text request it refuses to substitute commentary when the complete text cannot be verified.
- **Image upload with OCR.** Upload a photo of a poem (English or Hindi) and the agent reads it with Tesseract and identifies the work.
- **Related images with credits.** Wikimedia Commons and Unsplash images are shown with attribution and licence information.
- **Poem formatting.** Poems appear in their original line breaks, in a styled quote block.
- **Saved chats.** Conversations persist in PostgreSQL, with a chat sidebar and delete option.
- **Live progress.** The interface shows friendly status steps and streams the answer as it is written.
- **Tracing and feedback.** Runs are traced in LangSmith (route, tools used, whether the fallback fired) and each answer has a thumbs up/down.

---

## How it works

![Agent graph flowchart](screenshots/workflow.png)

The workflow is a fixed LangGraph graph with LLM-based decisions at key points (analysis, candidate selection, sufficiency check). Conversation state is checkpointed in PostgreSQL.

---

## Tech stack

| Area | Tools |
|------|-------|
| Orchestration | LangGraph, LangChain Core |
| Language model | DeepSeek (via the OpenAI-compatible API) |
| Interface | Streamlit |
| Storage | PostgreSQL (LangGraph checkpointer and chat list) |
| Retrieval | Wikisource, Wikipedia, Rekhta, Hindwi, Literary Hub, Tavily web search |
| Images | Wikimedia Commons, Unsplash |
| OCR | Tesseract (English and Hindi) |
| Tracing | LangSmith |

---

## Project structure

```
app.py          Streamlit interface, chat sidebar, streaming
graph.py        LangGraph workflow: analysis, routing, retrieval, generation
tools.py        Retrieval, OCR and image-search tools
prompts.py      System and analyzer prompts
thread.py       Chat list stored in PostgreSQL
configure.py    Model settings
.streamlit/     Theme configuration
```

---

## Setup

**Requirements:** Python 3.11+, a PostgreSQL database, and Tesseract OCR.

1. Clone the repository and create a virtual environment.

   ```bash
   git clone https://github.com/krskumarsatyam777-glitch/Literature-Discussion-Agent.git
   cd Literature-Discussion-Agent
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install Tesseract with English and Hindi data.

   ```bash
   sudo apt install tesseract-ocr tesseract-ocr-hin    # Debian/Ubuntu
   ```

   On Windows, install Tesseract and set `TESSERACT_CMD` to the path of `tesseract.exe`.

3. Copy `.env.example` to `.env` and fill in the values.

   ```env
   DEEPSEEK_API_KEY=
   DATABASE_URL=postgresql://user:password@host:5432/dbname
   TAVILY_API_KEY=
   UNSPLASH_ACCESS_KEY=
   TESSERACT_CMD=            # only if Tesseract is not on your PATH
   LANGSMITH_API_KEY=        # optional, for tracing and feedback
   ```

4. Run the app. The database tables are created automatically on first start.

   ```bash
   streamlit run app.py
   ```

---

## Example queries

- Explain Ghalib's ghazal *Hazaron Khwahishein Aisi*.
- Discuss the ending of *Hamlet*.
- Compare *Hamlet* and *Macbeth*.
- Give me the full text of *Ozymandias* and discuss it.
- What do critics say about *The Waste Land*?
- Upload a photo of a poem and ask what it is.

---

## Current limitations

- **Retrieval relies on hand-set rules.** Title matching, page-type detection and thresholds are heuristics that have not been tuned on a labelled dataset.
- **Wikipedia matching is unverified.** The first usable article is chosen, so an ambiguous name can occasionally return the wrong page.
- **Long works.** Plays and novels split across many Wikisource subpages are not yet followed page by page.
- **Source coverage is narrow.** Trusted literary sites (Rekhta, Hindwi, Literary Hub) are only searched for Hindi and Urdu queries; English relies on Wikisource, Wikipedia and web fallback.
- **Non-English queries are slower** because each language edition is searched in sequence.
- **OCR** supports English and Hindi only, and accuracy depends on image quality.
- **Chats are not per-user.** All saved chats share one list, so a public deployment would show everyone the same history.
- **No automated evaluation yet.** Answer quality has been checked manually.

---

## Future improvements

**Retrieval quality**
- Replace heuristic matching with semantic ranking (multilingual embeddings, optionally a cross-encoder reranker).
- Choose thresholds from a labelled set of queries and correct pages, instead of fixed values.
- Verify the selected Wikipedia article against the intended work or author.
- Follow index pages and subpages for multi-page works (acts, chapters).

**Coverage**
- Add more trusted literary sources and a per-language source configuration (for example Project Gutenberg, Poetry Foundation, Kavita Kosh).
- Expand supported languages, including OCR languages.

**Speed**
- Search language editions in parallel and cache repeated lookups.

**Evaluation**
- Offline evaluation on a fixed test set of queries.
- Online evaluators for hallucination, over-refusal and latency.

**Interface**
- Chat titles based on the work and author instead of the first message.
- Per-user chat history.

---

## Acknowledgements

LangGraph, LangChain, DeepSeek, Streamlit, Wikisource, Wikipedia, Wikimedia Commons, Rekhta, Hindwi, Literary Hub, Tavily, Unsplash and Tesseract OCR.