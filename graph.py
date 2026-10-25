import logging
import os
from typing import Annotated, Literal, TypedDict

import configure

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from prompts import ANALYZE_PROMPT, SYSTEM_PROMPT
from tools import (
    extract_text_from_upload,
    fetch_wikipedia_background,
    fetch_wikisource_page,
    image_mentions_author,
    looks_like_full_text_request,
    make_result,
    search_literary_sites,
    search_unsplash_images,
    search_web,
    search_wikimedia_images,
    search_wikisource_pages,
    unsplash_scene_relevant,
    work_image_relevant,
    work_title_match_score,
)

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver


log = logging.getLogger(__name__)


# ====================================LLM=====================================================

llm = ChatOpenAI(
    model=getattr(configure, "LLM_MODEL", "deepseek-v4-flash"),
    api_key=(
        getattr(configure, "DEEPSEEK_API_KEY", None)
        or os.environ["DEEPSEEK_API_KEY"]
    ),
    base_url=getattr(
        configure, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    ),
    temperature=0,
)


# ======================================STATE====================================================

def merge_results(old, new):
    """Append new tool results. An empty list clears them; begin() sends one at
    the start of every turn, so results never leak between turns."""
    if not new:
        return []
    return (old or []) + new


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    # Input. Pass image_path on every turn (None when there is no image).
    image_path: str
    ocr_text: str
    ocr_failed: bool

    # Filled by analyze
    analysis_failed: bool
    in_scope: bool
    route: str
    entities: list[str]
    language: str
    search_query: str
    original_query: str
    work_title: str
    needs_full_text: bool
    needs_image: bool
    image_query: str
    wikimedia_query: str
    wants_opinions: bool
    needs_retrieval: bool

    # Filled by the retrieval and image nodes (reset every turn)
    results: Annotated[list, merge_results]
    images: list

    # Kept between turns for follow-ups and same-author context reuse
    last_results: list
    author: str
    last_author: str
    reuse_wikipedia: bool
    last_wikipedia_result: dict | None


#==============================Analyzer (one structured LLM call)======================================

class Analysis(BaseModel):
    in_scope: bool = Field(
        description="True for discussion, interpretation, comparison, reading, or "
        "research about literary works and authors across forms: poems, ghazals, "
        "nazms, short stories, novels, novellas, plays, letters, essays, speeches, "
        "diaries, memoirs, autobiographies, fables, myths, folklore, literary "
        "criticism, translation, and literary movements. False for unrelated requests "
        "such as sports, coding help, general chat, or image-generation requests."
    )
    route: Literal["new_topic", "related", "follow_up"] = Field(
        description="new_topic: a different work or author than the recent "
        "conversation. related: a different work but the same author, theme "
        "or tradition. follow_up: a question about the work already being "
        "discussed. Use new_topic if unsure or if there is no earlier discussion."
    )
    entities: list[str] = Field(
        description="Author and work names the user is asking about, in the "
        "spelling most likely used by Wikisource (for example 'Hamlet', "
        "'William Shakespeare'). Empty list if none."
    )
    author: str = Field(
        description="The primary author of the literary work currently being "
        "discussed, written as the name they are most commonly known by (the "
        "name used as their Wikipedia article title or on a book cover). "
        "Examples: 'Mark Twain' not 'Samuel Clemens', 'Premchand', "
        "'Rabindranath Tagore', 'W. B. Yeats'. Leave out middle names and "
        "titles unless they are normally part of the name. Return an empty "
        "string if the author cannot be identified."
    )
    work_title: str = Field(
        description="The specific literary work title without task instructions; "
        "empty string if no work is identified."
    )
    needs_full_text: bool = Field(
        description="True when the user asks for the actual or complete text of "
        "a literary work; false for analysis, biography, themes, or discussion."
    )
    language: str = Field(
        description="Language code of the work's original language, used to "
        "choose the Wikisource edition: en, hi, ur, bn, ta, and so on. "
        "Use en if unsure."
    )
    needs_image: bool = Field(
        description="True only when a relevant image would materially help for a "
        "new work or an author-focused request. For a specific work, prefer its "
        "setting, symbols, artefacts, or documented illustration; do not request "
        "an author portrait by default. False for routine follow-ups."
    )
    image_query: str = Field(
        description="A specific, concrete visual scene for stock photography, "
        "not merely an abstract emotion. Empty if no image is needed."
    )
    wikimedia_query: str = Field(
        description="Specific Wikimedia Commons search terms for an identifiable "
        "image related to the work or author, such as a portrait, manuscript, or "
        "known illustration. Empty if no image is needed."
    )
    wants_opinions: bool = Field(
        description="True only if the user explicitly asks for reviews, "
        "opinions or reader reactions."
    )
    needs_retrieval: bool = Field(
        description="False only for a follow_up that can be answered from the "
        "text already discussed. True otherwise."
    )

# DeepSeek runs in thinking mode by default, and thinking mode rejects the forced
# tool_choice that structured output uses (400 "does not support this tool_choice").
# The analyzer is a simple classification, so it runs with thinking switched off.
analyzer_llm = ChatOpenAI(
    model=getattr(configure, "LLM_MODEL", "deepseek-v4-flash"),
    api_key=(
        getattr(configure, "DEEPSEEK_API_KEY", None)
        or os.environ["DEEPSEEK_API_KEY"]
    ),
    base_url=getattr(
        configure, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    ),
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)
analyzer = analyzer_llm.with_structured_output(Analysis, method="function_calling")


class CandidateSelection(BaseModel):
    indices: list[int] = Field(
        description="Candidate indices ordered from best to worst match, at most 3. "
        "Use only supplied indices and prefer a direct page for the requested work."
    )


class RetrievalAssessment(BaseModel):
    adequate: bool = Field(
        description="True only if the retrieved source text is relevant and sufficient "
        "for the user's request. For a full-text request, require the actual complete "
        "work, not an excerpt, snippet, author page, or truncated text."
    )
    reason: str = Field(description="Brief reason for the decision.")

candidate_selector = analyzer_llm.with_structured_output(
    CandidateSelection, method="function_calling"
)
retrieval_evaluator = analyzer_llm.with_structured_output(
    RetrievalAssessment, method="function_calling"
)

MAX_OCR_CHARS_FOR_ANALYSIS = 1500

REFUSE_MESSAGE = (
    "Sorry, I can only help with literature-related topics, including poems, "
    "stories, novels, plays, letters, essays, authors, literary themes, and "
    "interpretation of literary works."
)
OCR_FAILED_MESSAGE = (
    "I couldn't read any text from that image. Could you upload a clearer "
    "photo or screenshot?"
)
ANALYSIS_FAILED_MESSAGE = (
    "Something went wrong while reading that message. Please try sending it again."
)


# ============================Fixed replies ================================================

def reply_node(message: str):
    """Build a node that answers with a fixed message and ends the turn."""
    def node(state: State):
        return {"messages": [AIMessage(content=message)]}
    return node


# =========================Turn Initialization and OCR Processing=================================

def begin(state: State):
    """Reset everything that belongs to a single turn."""
    return {
        "results": [],
        "images": [],
        "ocr_text": "",
        "ocr_failed": False,
        "analysis_failed": False,
    }


def route_start(state: State):
    return "ocr" if state.get("image_path") else "analyze"


def ocr(state: State):
    """Extract text safely; OCR errors should lead to the fixed failure reply."""
    try:
        results = extract_text_from_upload(state["image_path"])
        result = results[0] if results else {}
        text = (
            (result.get("text") or "").strip()
            if result.get("status") == "ok"
            else ""
        )
    except Exception:
        log.exception("OCR failed")
        text = ""

    return {"ocr_text": text, "ocr_failed": not bool(text)}


def after_ocr(state: State):
    return "ocr_failed" if state.get("ocr_failed") else "analyze"


# ================================Analyze, scope guard, routing =======================================

def analyze(state: State):
    messages = state["messages"]
    transcript = "\n".join(f"{m.type}: {m.content}" for m in messages[-6:])
    if state.get("ocr_text"):
        transcript += (
            "\n\n[Text read from the uploaded image]\n"
            + state["ocr_text"][:MAX_OCR_CHARS_FOR_ANALYSIS]
        )

    # Give the analyzer explicit routing guidance and the previous author.
    previous_author = state.get("last_author") or "unknown"
    analyzer_system = (
        ANALYZE_PROMPT
        + "\n\nRouting safeguards: if there is no earlier discussion, "
        "or if the route is uncertain, choose new_topic. Use related for a "
        "different work connected by author, theme, or tradition. Use follow_up "
        "only when the user is asking about the work already under discussion. "
        f"\nPrevious identified author: {previous_author}."
    )

    analysis = None
    for _ in range(2):  # one retry
        try:
            analysis = analyzer.invoke(
                [SystemMessage(analyzer_system), HumanMessage(transcript)]
            )
        except Exception as e:
            log.warning("analyze call failed: %s", e)
        if analysis:
            break

    if analysis is None:
        return {"analysis_failed": True}

    data = analysis.model_dump()

    # Enforce routing defaults in code where possible.
    if state.get("image_path") or len(messages) == 1:
        data["route"] = "new_topic"
    if data.get("route") not in {"new_topic", "related", "follow_up"}:
        data["route"] = "new_topic"
    if data["route"] != "follow_up":
        data["needs_retrieval"] = True

    # Preserve the user's wording for targeted searches and fallback.
    original_query = str(messages[-1].content).strip()
    data["original_query"] = original_query
    data["needs_full_text"] = bool(
        data.get("needs_full_text") or looks_like_full_text_request(original_query)
    )
    ocr_snippet = (state.get("ocr_text") or "")[:300].strip()
    data["search_query"] = (
        " ".join(data.get("entities") or [])
        or ocr_snippet
        or original_query
    )

    current_author = (data.get("author") or "").strip().casefold()
    previous_author_normalized = previous_author.strip().casefold()
    data["reuse_wikipedia"] = bool(
        data["route"] == "related"
        and current_author
        and previous_author_normalized != "unknown"
        and current_author == previous_author_normalized
        and state.get("last_wikipedia_result")
    )

    data["analysis_failed"] = False
    return data


def route_after_analyze(state: State):
    """Scope guard first, then the route decides which nodes run."""
    if state.get("analysis_failed"):
        return "analysis_error"
    if not state.get("in_scope"):
        return "refuse"

    route = state.get("route")

    if route == "follow_up" and not state.get("needs_retrieval"):
        return "generate"

    nodes = ["wikisource", "wikipedia"]
    
    if (state.get("language") or "en").lower() in ("ur", "hi"):
        nodes.append("literary_sites")
    if state.get("wants_opinions"):
        nodes.append("web_opinions")
    return nodes


# =================================== Retrieval ======================================================

def _unique_queries(values):
    """Keep nonempty search queries while removing duplicates."""
    seen, queries = set(), []
    for value in values:
        value = " ".join(str(value or "").split()).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            queries.append(value)
    return queries


def _candidate_score(candidate, work_title, author):
    """Simple backup ranking if the candidate-selection model fails."""
    title = candidate.get("title", "").replace("_", " ").casefold()
    snippet = candidate.get("snippet", "").casefold()
    text = f"{title} {snippet}"
    score = 5 if work_title and work_title.casefold() in title else 0
    score += sum(2 for word in (work_title or "").casefold().split()
                 if len(word) > 2 and word in text)
    score += sum(1 for word in (author or "").casefold().split()
                 if len(word) > 2 and word in text)
    return score


def wikisource(state: State):
    """Search candidate pages, select likely matches, then fetch exact titles."""
    work_title = (state.get("work_title") or "").strip()
    author = (state.get("author") or "").strip()
    original_query = state.get("original_query") or state.get("search_query") or ""
    lang = (state.get("language") or "en").lower()
    languages = _unique_queries([lang, "en"])
    entity_query = " ".join(state.get("entities") or [])
    queries = _unique_queries([
        work_title,
        f"{work_title} {author}" if work_title and author else "",
        original_query,
        entity_query,
    ])[:3]

    candidates, seen = [], set()
    for search_lang in languages:
        language_count = 0
        for query in queries:
            for candidate in search_wikisource_pages(query, lang=search_lang, limit=5):
                key = (search_lang, candidate.get("title", "").casefold())
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
                    language_count += 1
                if language_count >= 8:
                    break
            if language_count >= 8:
                break

    if not candidates:
        return {"results": [make_result(
            "wikisource", 1, "not_found", title=work_title or original_query
        )]}

    # Fast path: if a candidate title/URL clearly matches the requested work,
    # fetch that page directly instead of paying for a separate selector LLM call.
    # Keep model selection for ambiguous cases (including many transliterations).
    ranked_by_title = sorted(
        candidates,
        key=lambda item: work_title_match_score(item, work_title),
        reverse=True,
    ) if work_title else []
    best_title_score = (
        work_title_match_score(ranked_by_title[0], work_title)
        if ranked_by_title else 0.0
    )
    selection_succeeded = False
    if best_title_score >= 0.90:
        selected = [ranked_by_title[0]]
        selection_succeeded = True
    else:
        candidate_text = "\n\n".join(
            f"[{i}] {item.get('title', '')} | {item.get('url', '')}\n"
            f"Snippet: {item.get('snippet', '')}"
            for i, item in enumerate(candidates)
        )
        try:
            selection = candidate_selector.invoke([
                SystemMessage(content=(
                    "Select up to three Wikisource candidates that best match the "
                    "specific requested work. Prefer the actual text page, not an "
                    "author biography, contents page, collection, or unrelated work. "
                    "Return only candidate indices; return an empty list if none match."
                )),
                HumanMessage(content=(
                    f"Request: {original_query}\nWork: {work_title or 'unknown'}\n"
                    f"Author: {author or 'unknown'}\n\n{candidate_text}"
                )),
            ])
            selection_succeeded = True
            selected = [candidates[i] for i in selection.indices
                        if isinstance(i, int) and 0 <= i < len(candidates)][:3]
        except Exception:
            log.exception("Wikisource selection failed; using title matching")
            selected = []

    # Only use the deterministic fallback when the model call failed. An empty
    # selection means the model found no plausible candidate; let web fallback run.
    if not selected and not selection_succeeded:
        ranked = sorted(
            candidates,
            key=lambda item: _candidate_score(item, work_title, author),
            reverse=True,
        )
        if ranked and _candidate_score(ranked[0], work_title, author) > 0:
            selected = ranked[:3]

    fetched = []
    for candidate in selected:
        result = fetch_wikisource_page(
            candidate["title"], lang=candidate.get("lang") or lang
        )
        result["search_snippet"] = candidate.get("snippet", "")
        fetched.append(result)

    return {"results": fetched or [make_result(
        "wikisource", 1, "not_found", title=work_title or original_query
    )]}


def wikipedia(state: State):
    """Reuse cached author background for a related work by the same author."""
    cached = state.get("last_wikipedia_result")
    if state.get("reuse_wikipedia") and cached:
        return {"results": [cached]}
    return {"results": fetch_wikipedia_background(
        state["search_query"], lang=state.get("language") or "en"
    )}


def literary_sites(state: State):
    if state.get("needs_full_text"):
        query = " ".join(part for part in (
            state.get("work_title") or "",
            state.get("author") or "",
            "full text",
        ) if part).strip()
        if not state.get("work_title"):
            query = state.get("original_query") or state["search_query"]
    else:
        query = state.get("original_query") or state["search_query"]
    return {
        "results": search_literary_sites(
            query,
            max_results=5 if state.get("needs_full_text") else 3,
            fetch_full_page=bool(state.get("needs_full_text")),
        )
    }


def web_opinions(state: State):
    query = f"{state['search_query']} reader reviews criticism"
    results = search_web(query)
    for result in results:
        result["purpose"] = "opinions"
    return {"results": results}


def _parse_meta(text):
    return dict(
        line.split(": ", 1) for line in (text or "").splitlines() if ": " in line
    )


def _image_record(result, source):
    meta = _parse_meta(result.get("text", ""))
    if source == "wikimedia":
        return {
            "source": source,
            "url": result["url"],
            "title": result.get("title", ""),
            "credit_name": meta.get("Artist", "Unknown"),
            "credit_url": meta.get("Image page", ""),
            "page_url": meta.get("Image page", ""),
            "license": meta.get("License", ""),
            "license_url": meta.get("License URL", ""),
            "download_location": "",
        }
    return {
        "source": source,
        "url": result["url"],
        "title": result.get("title", ""),
        "credit_name": meta.get("Photographer", "Unknown"),
        "credit_url": meta.get("Photographer profile", ""),
        "page_url": meta.get("Image page", ""),
        "license": "Unsplash License",
        "license_url": "",
        "download_location": meta.get("Download tracking URL", ""),
    }



def _author_focused_request(state):
    query = " ".join(
        str(state.get("original_query", "")).casefold()
        .replace(".", " ").replace(",", " ").split()
    )
    work_title = (state.get("work_title") or "").strip()
    explicit_author_request = any(phrase in query for phrase in (
        "biography", "life of", "portrait of", "photo of", "photograph of",
        "picture of", "image of", "who was", "who is", "author portrait",
        "writer portrait", "poet portrait", "tell me about the author",
    ))
    if work_title and not explicit_author_request:
        return False
    return bool(state.get("author")) and (not work_title or explicit_author_request)


def fetch_images(state: State):
    """Prefer work-relevant imagery; show portraits only for author-focused requests."""
    try:
        work_title = (state.get("work_title") or "").strip()
        author = (state.get("author") or "").strip()
        theme = (state.get("image_query") or "").strip()
        commons_query = (state.get("wikimedia_query") or "").strip()
        author_focus = _author_focused_request(state)
        images, used_urls = [], set()

        def add_candidate(result, source):
            url = result.get("url")
            if result.get("status") != "ok" or not url or url in used_urls:
                return False
            used_urls.add(url)
            images.append(_image_record(result, source))
            return True

        if author_focus and author:
            for query in _unique_queries([
                f'"{author}" portrait', f'"{author}" photograph',
            ]):
                for result in search_wikimedia_images(query, per_page=8):
                    if image_mentions_author(result, author) and add_candidate(result, "wikimedia"):
                        break
                if images:
                    break
            return {"images": images[:1]}

        queries = _unique_queries([
            commons_query,
            f'"{work_title}" illustration' if work_title else "",
            f'"{work_title}" painting' if work_title else "",
            theme,
        ])[:3]
        for query in queries:
            for result in search_wikimedia_images(query, per_page=8):
                if work_image_relevant(result, work_title, theme):
                    add_candidate(result, "wikimedia")
                if len(images) >= 2:
                    break
            if len(images) >= 2:
                break

        # Unsplash is a thematic fallback/companion only when its metadata supports the query.
        if theme and len(images) < 3:
            for result in search_unsplash_images(theme, per_page=8):
                if unsplash_scene_relevant(result, theme):
                    add_candidate(result, "unsplash")
                    break
        return {"images": images}
    except Exception:
        log.exception("Image fetch failed")
        return {"images": []}


def evaluate(state: State):
    return {}


def _retrieval_is_adequate(state: State) -> bool:
    """Use a fast path for clearly relevant discussion sources; judge ambiguous cases."""
    retrieval_results = list(state.get("results") or [])

    # For ordinary discussion (not complete-text requests), avoid an extra model
    # call when a trusted page directly matches the named work and has substantial
    # content. Full-text requests always retain the stricter completeness evaluator.
    if (
        not state.get("needs_full_text")
        and not state.get("wants_opinions")
        and state.get("work_title")
    ):
        for result in retrieval_results:
            if (
                result.get("status") == "ok"
                and result.get("source") in {"wikisource", "wikipedia", "literary_sites"}
                and len((result.get("text") or "").strip()) >= 250
                and work_title_match_score(result, state["work_title"]) >= 0.90
            ):
                log.info("Retrieval fast path: relevant page matched the requested work")
                return True

    sources = []
    if state.get("needs_full_text") and state.get("work_title"):
        retrieval_results.sort(
            key=lambda item: work_title_match_score(
                item, state.get("work_title") or ""
            ),
            reverse=True,
        )
    for result in retrieval_results:
        if result.get("status") != "ok" or not (result.get("text") or "").strip():
            continue
        if result.get("source") not in {
            "wikisource", "wikipedia", "literary_sites", "web_search"
        }:
            continue
        if result.get("purpose") == "opinions" and not state.get("wants_opinions"):
            continue
        text = result["text"].strip()
        if len(text) > 4200:
            text = text[:3000] + "\n[Middle omitted for evaluation]\n" + text[-1000:]
        sources.append(
            f"SOURCE: {result.get('source')} | {result.get('title', '')} | {result.get('url', '')}\n"
            f"TRUNCATED: {bool(result.get('truncated'))}\nTEXT:\n{text}"
        )
        if len(sources) >= 6:
            break

    if not sources:
        return False

    system = (
        "Judge whether these retrieved sources are sufficient for the user's actual "
        "literary request. If needs_full_text is true, mark adequate only if a source "
        "contains the actual complete requested work; a snippet, opening couplet, "
        "biography, catalogue, or truncated source is not enough. Otherwise, mark "
        "adequate only if the sources genuinely help answer the question. A biography "
        "does not answer a question about a specific poem. If uncertain, say false. "
        "The text displayed to you may omit its middle only for evaluation prompt size; "
        "use TRUNCATED to determine whether the actual fetched source was clipped. "
        "Do not use outside knowledge to fill missing text."
    )

    #user prompt
    user = (
        f"Request: {state.get('original_query') or state.get('search_query', '')}\n"
        f"Work: {state.get('work_title') or 'unknown'}\n"
        f"Author: {state.get('author') or 'unknown'}\n"
        f"Needs complete text: {bool(state.get('needs_full_text'))}\n"
        f"Wants reviews/opinions: {bool(state.get('wants_opinions'))}\n\n"
        + "\n\n---\n\n".join(sources)
    )
    try:
        assessment = retrieval_evaluator.invoke([
            SystemMessage(content=system), HumanMessage(content=user)
        ])
        log.info("Retrieval adequate=%s: %s", assessment.adequate, assessment.reason)
        return bool(assessment.adequate)
    except Exception:
        log.exception("Retrieval evaluation failed; triggering web fallback")
        return False


def _should_fetch_images(state: State) -> bool:
    return bool(state.get("needs_image") and state.get("route") in ("new_topic", "related"))


def after_evaluate(state: State):
    if not _retrieval_is_adequate(state):
        return "web_fallback"
    return "fetch_images" if _should_fetch_images(state) else "generate"


def web_fallback(state: State):
    if state.get("needs_full_text"):
        query = " ".join(part for part in (
            state.get("work_title") or "",
            state.get("author") or "",
            "full text",
        ) if part).strip()
        if not state.get("work_title"):
            query = state.get("original_query") or state.get("search_query") or ""
    elif state.get("wants_opinions"):
        query = f"{state.get('original_query') or state['search_query']} reviews criticism"
    else:
        query = state.get("original_query") or state["search_query"]
    return {
        "results": search_web(
            query,
            max_results=5 if state.get("needs_full_text") else 3,
            fetch_full_page=bool(state.get("needs_full_text")),
        )
    }


def after_fallback(state: State):
    # Do not substitute commentary for a full-text request when every source is
    # still an excerpt or an unrelated page after the broader web search.
    if state.get("needs_full_text") and not _retrieval_is_adequate(state):
        return "full_text_unavailable"
    return "fetch_images" if _should_fetch_images(state) else "generate"


#=====================================Answer ========================================================

def format_sources(results):
    blocks = []
    for r in results or []:
        if r["status"] == "ok" and r["text"]:
            reliability = "trusted" if r["tier"] == 1 else "open web, unverified"
            blocks.append(
                f"[SOURCE: {r['source']} | {reliability} | {r['title']} | {r['url']}]\n"
                f"{r['text']}"
            )
    return "\n\n".join(blocks) or "NO SOURCES WERE RETRIEVED."


def generate(state: State):
    usable = [
        r
        for r in (state.get("results") or [])
        if r["status"] == "ok" and r["text"]
    ]

    # For a full-text request, prefer source pages whose titles/URLs match the
    # requested work. This prevents an author's unrelated poem from being quoted
    # just because its page appeared earlier in search results.
    if state.get("needs_full_text") and state.get("work_title"):
        matching_work = [
            item for item in usable
            if work_title_match_score(item, state.get("work_title") or "") >= 0.5
        ]
        if matching_work:
            usable = matching_work

    # The uploaded image's text is the user's own material; give it to the model.
    if state.get("ocr_text"):
        usable.insert(
            0,
            make_result(
                "uploaded_image",
                1,
                "ok",
                title="Uploaded image (OCR text)",
                text=state["ocr_text"],
            ),
        )

    # A follow-up that skipped retrieval (or found nothing new) reuses the
    # sources from the previous turn.
    if not usable and state.get("route") == "follow_up":
        usable = state.get("last_results") or []

    request_mode = ""
    if state.get("needs_full_text"):
        request_mode = (
            "\n\nCURRENT REQUEST MODE: FULL-TEXT REQUEST. If one of the sources "
            "contains the complete requested work, give that text first, "
            "preserving its original line breaks, then add a short discussion of "
            "the poem. If you only have part of the text, give what you have, say "
            "so in one short sentence, and then discuss the work. Never fill gaps "
            "from memory or with lines from other works."
        )
    system = (
        SYSTEM_PROMPT + request_mode + "\n\nRETRIEVED SOURCES:\n"
        + format_sources(usable)
    )
    response = llm.invoke([SystemMessage(system)] + state["messages"])

    # Keep only a usable Wikipedia result for the next turn. Since a reused
    # result is also in `usable`, it is retained without another API call.
    wikipedia_result = next(
        (
            result for result in usable
            if result.get("source") == "wikipedia"
            and result.get("status") == "ok"
            and result.get("text")
        ),
        None,
    )

    current_author = (state.get("author") or "").strip()
    if state.get("route") == "follow_up" and not current_author:
        next_author = state.get("last_author", "")
    else:
        next_author = current_author

    return {
        "messages": [response],
        "last_results": usable,
        "last_author": next_author,
        "last_wikipedia_result": wikipedia_result,
    }


# ==========================================Graph ====================================================

builder = StateGraph(State)

builder.add_node("begin", begin)
builder.add_node("ocr", ocr)
builder.add_node("ocr_failed", reply_node(OCR_FAILED_MESSAGE))
builder.add_node("analyze", analyze)
builder.add_node("analysis_error", reply_node(ANALYSIS_FAILED_MESSAGE))
builder.add_node("refuse", reply_node(REFUSE_MESSAGE))
builder.add_node("wikisource", wikisource)
builder.add_node("wikipedia", wikipedia)
builder.add_node("literary_sites", literary_sites)
builder.add_node("web_opinions", web_opinions)
builder.add_node("fetch_images", fetch_images)
builder.add_node("evaluate", evaluate)
builder.add_node("web_fallback", web_fallback)
builder.add_node(
    "full_text_unavailable",
    reply_node(
        "I couldn't verify the complete text of this work from the pages I could "
        "retrieve. I won't substitute an excerpt or interpretation for the full text."
    ),
)
builder.add_node("generate", generate)

builder.add_edge(START, "begin")
builder.add_conditional_edges("begin", route_start, ["ocr", "analyze"])
builder.add_conditional_edges("ocr", after_ocr, ["ocr_failed", "analyze"])
builder.add_edge("ocr_failed", END)

# Scope guard and route: returns one node name or a list of nodes to run in parallel.
PARALLEL_NODES = [
    "wikisource",
    "wikipedia",
    "literary_sites",
    "web_opinions",
]
builder.add_conditional_edges(
    "analyze",
    route_after_analyze,
    ["analysis_error", "refuse", "generate"] + PARALLEL_NODES,
)
builder.add_edge("analysis_error", END)
builder.add_edge("refuse", END)

# Each parallel node reports to evaluate, which runs once they have all finished.
for node in PARALLEL_NODES:
    builder.add_edge(node, "evaluate")

builder.add_conditional_edges(
    "evaluate", after_evaluate, ["web_fallback", "fetch_images", "generate"]
)
builder.add_conditional_edges(
    "web_fallback",
    after_fallback,
    ["full_text_unavailable", "fetch_images", "generate"],
)
builder.add_edge("full_text_unavailable", END)
builder.add_edge("fetch_images", "generate")
builder.add_edge("generate", END)

pool = ConnectionPool(
    conninfo=os.environ["DATABASE_URL"],
    max_size=5,
    open=True,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # creates the checkpoint tables; safe to run on every start

graph = builder.compile(checkpointer=checkpointer)
