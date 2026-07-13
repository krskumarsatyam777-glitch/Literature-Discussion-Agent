# Literature Discussion Agent

An AI-powered Literature Discussion Agent built using **LangChain**, **DeepSeek**, and **Agentic Retrieval-Augmented Generation (RAG)** to provide grounded discussions on poems, novels, authors, literary themes, and literary criticism.

Unlike a traditional chatbot that answers solely from the language model's internal knowledge, this agent dynamically retrieves information from trusted literary sources before generating responses. It combines retrieval, OCR, web search, thematic image retrieval, and literary recommendations within a single LangChain agent.

---

# Features

- Agentic Retrieval-Augmented Generation (RAG)
- LangChain Agent with dynamic tool selection
- OCR support for uploaded literary images
- Full literary work retrieval from Wikisource
- Author information retrieval from Wikipedia
- Web search integration through Tavily
- Thematic image retrieval using the Unsplash API
- Literary recommendation system
- Automatic Reading Diary generation
- Chat export
- Reading Diary export
- Interactive Streamlit interface

---
## Live Demo

**Application:** https://literature-discussion-agent.onrender.com

**GitHub Repository:** https://github.com/krskumarsatyam777-glitch/Literature-Discussion-Agent
---

# Technologies Used

## Framework

- LangChain
- Streamlit

## Language Model

- DeepSeek V4 Flash API

## Retrieval Sources

- Wikisource
- Wikipedia
- Tavily Search API

## OCR

- Tesseract OCR

## Image Retrieval

- Unsplash API

## Programming Language

- Python

---

# Architecture

```
                    User Query
                         │
                         ▼
                 Streamlit Interface
                         │
                         ▼
                 LangChain Agent
                         │
        Dynamically selects required tools
                         │
 ┌───────────────┬──────────────┬───────────────┐
 │               │              │               │
 ▼               ▼              ▼               ▼
OCR        Wikisource     Wikipedia      Tavily Search
 │               │              │               │
 └───────────────┴──────────────┴───────────────┘
                         │
          Retrieved Literary Context
                         │
                         ▼
                DeepSeek V4 Flash
                         │
                         ▼
      Literary Discussion + Recommendations
         + Reading Diary + Thematic Images
```

---

# Agentic Retrieval-Augmented Generation (RAG)

This project implements **Agentic RAG** rather than classical vector-database RAG.

Instead of retrieving passages from embeddings stored in a vector database, the LangChain Agent dynamically decides which external tools should be used based on the user's request.

Depending on the query, the agent may retrieve:

- Complete literary works from Wikisource
- Author information from Wikipedia
- Additional literary context through Tavily Search
- Text extracted from uploaded images using OCR
- Thematic images related to literary works
- Similar literary recommendations

The retrieved information is then supplied to the DeepSeek language model, allowing it to generate responses grounded in external sources rather than relying solely on its training data. This significantly reduces hallucinations while enabling richer literary discussions.

---

# LangChain Agent Workflow

The project is built using a **LangChain Agent** instead of a fixed sequential pipeline.

Rather than executing the same sequence of operations for every query, the agent reasons about the user's request and dynamically determines which tools are necessary.

For example:

### Poem Analysis

```
User Query
      │
      ▼
Search Wikisource
      │
      ▼
Retrieve Complete Poem
      │
      ▼
Retrieve Author Information
      │
      ▼
Retrieve Thematic Images
      │
      ▼
Generate Literary Analysis
```

### Author Query

```
User Query
      │
      ▼
Retrieve Wikipedia Information
      │
      ▼
Generate Response
```

### Uploaded Literary Image

```
User Upload
      │
      ▼
OCR
      │
      ▼
Identify Literary Work
      │
      ▼
Retrieve Text
      │
      ▼
Generate Discussion
```

This dynamic tool selection allows different requests to follow different execution paths while using the same agent.

---

# LangChain Components

The project uses several LangChain components to orchestrate the workflow.

- ChatOpenAI interface configured for the DeepSeek API
- LangChain Agent for reasoning and tool selection
- Custom LangChain Tools
- Prompt Templates
- Runnable Chains
- Output Parser

---

# Custom Tools

The application implements multiple custom LangChain tools.

| Tool | Purpose |
|------|---------|
| OCR Tool | Extract text from uploaded literary images using Tesseract OCR |
| Wikisource Search Tool | Locate literary works |
| Wikisource Retrieval Tool | Retrieve complete literary texts |
| Wikipedia Tool | Retrieve author biographies |
| Tavily Search Tool | Retrieve additional literary context |
| Recommendation Tool | Recommend similar literary works |
| Thematic Image Tool | Retrieve thematic images from Unsplash |

---

# Prompt Engineering

The system prompt guides the agent to:

- Retrieve literary works before discussing them.
- Clearly distinguish retrieved facts from interpretation.
- Avoid inventing quotations or literary facts.
- Perform deeper thematic and symbolic analysis after retrieval.
- Use OCR automatically when literary images are uploaded.
- Retrieve thematic images only when appropriate.

---

# Reading Diary

Each interaction automatically generates a structured Reading Diary entry containing:

- Literary work discussed
- Author
- Themes
- Symbols
- Summary of discussion

The Reading Diary can be exported independently of the chat history.

---

# Project Structure

```
Literature-Discussion-Agent/

│── app.py
│── agent.py
│── tools.py
│── parser.py
│── prompts.py
│── configure.py
│── utils.py
│── requirements.txt
│── README.md

├── chat_logs/
├── reading_diary/
├── notebooks/
│      development_and_testing.ipynb
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/Literature-Discussion-Agent.git

cd Literature-Discussion-Agent
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```env
DEEPSEEK_API_KEY=your_key
TAVILY_API_KEY=your_key
UNSPLASH_ACCESS_KEY=your_key
```

Run the application.

```bash
streamlit run app.py
```

---

# Example Queries

- Analyze *The Road Not Taken*.
- Explain the symbolism in *The Waste Land*.
- Compare Hamlet and Macbeth.
- Explain Ghalib's poetry.
- Discuss existentialism in Kafka's works.
- Recommend novels similar to *The Trial*.
- Upload an image of a poem for identification and discussion.

---

# Development Notebook

The repository includes a development notebook documenting the incremental development of the project, including:

- LangChain tool development
- Prompt engineering
- OCR implementation
- Wikisource parsing
- Agent testing
- Migration from Gemini API to DeepSeek API
- Debugging and experimentation

---

# Current Limitations

- Response generation may take between 20–60 seconds depending on retrieval complexity.
- Hindi literary retrieval is generally slower than English.
- OCR accuracy depends on image quality.
- Thematic image retrieval depends on the availability and relevance of Unsplash results.

---

# Future Improvements

- Reduce response latency through retrieval and agent optimization.
- Improve multilingual support with additional Indian and international literary sources.
- Integrate additional literature databases.
- Enhance the Streamlit interface and overall user experience.
- Improve thematic image retrieval and recommendation quality.

---

# Acknowledgements

This project uses the following open-source frameworks and APIs:

- LangChain
- DeepSeek API
- Streamlit
- Wikisource
- Wikipedia
- Tavily Search API
- Unsplash API
- Tesseract OCR
