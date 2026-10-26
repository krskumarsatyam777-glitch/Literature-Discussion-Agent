import logging
import os
import re
import urllib.parse
from difflib import SequenceMatcher

import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tavily import TavilyClient

_tesseract_cmd = os.getenv("TESSERACT_CMD")

if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

# ===============================HEADER=========================================================

HEADERS = {
    "User-Agent": (
        "LiteratureDiscussionAgent "
        "(https://github.com/krskumarsatyam777-glitch/"
        "Literature-Discussion-Agent)"
    )
}


# ===============================HELPER FUNCTIONS (SHARED)=========================================

def get_json(url, params, headers=None, retries=1, timeout=10):
    """Fetch JSON with a timeout and one retry."""
    merged = {**HEADERS, **(headers or {})}
    last_error = None

    for _ in range(retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=merged,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e

    raise last_error


def make_result(
    source, tier, status, title="", url="", text="", error="", **extra
):
    """Create a consistent result dictionary. Extra keyword arguments are added as additional keys."""
    return {
        "source": source,
        "tier": tier,
        "status": status,
        "title": title,
        "url": url,
        "text": text,
        "error": error,
        **extra,
    }

_tavily_client = None
def _get_tavily():
    """Create the Tavily client on first use, so the key is read at call time."""
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _clean_lang(lang):
    """Accept only a plain language code (en, hi, ur, zh-hans); else use 'en'."""
    if re.fullmatch(r"[a-z]{2,3}(-[a-z]{2,8})?", lang or ""):
        return lang
    return "en"


def _strip_html(value):
    """Turn an HTML snippet (e.g. a Commons Artist field) into plain text."""
    return BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)


def looks_like_full_text_request(query):
    """Deterministically catch common requests for the actual complete work."""
    query = " ".join((query or "").casefold().split())
    patterns = (
        r"\b(full|complete|entire|whole)\s+(text|poem|ghazal|nazm|work|lyrics)\b",
        r"\b(give|show|provide|share|send|print|write out)\s+(me\s+)?"
        r"(the\s+)?(full\s+|complete\s+|entire\s+)?"
        r"(text|poem|ghazal|nazm|lyrics)\b",
        r"\b(text|poem|ghazal|nazm|lyrics)\s+(itself|in full|in its entirety)\b",
        r"\binclude\s+(the\s+)?(original\s+)?(full\s+)?poem\b",
        r"\bwhat (is|are) the (full )?text of\b",
        r"\ball (the )?(poem )?(lines|verses)\b",
    )
    return any(re.search(pattern, query) for pattern in patterns)


def work_title_match_score(result, work_title):
    """Compare source title/URL to a work title, tolerating transliteration."""
    if not work_title:
        return 0.0
    title = result.get("title") or ""
    path = urllib.parse.unquote(
        urllib.parse.urlparse(result.get("url") or "").path.replace("-", " ")
    )
    haystack = re.sub(r"[^a-z0-9]+", " ", f"{title} {path}".casefold()).split()
    target = re.sub(r"[^a-z0-9]+", " ", work_title.casefold()).split()
    target = [token for token in target if len(token) > 2]
    if not target or not haystack:
        return 0.0
    matched = sum(
        any(
            token == candidate
            or (min(len(token), len(candidate)) >= 4
                and SequenceMatcher(None, token, candidate).ratio() >= 0.82)
            for candidate in haystack
        )
        for token in target
    )
    return matched / len(target)


# ===============================TOOL 1: WIKISOURCE=================================================

MAX_WIKISOURCE_CHARS = 10000


# ----------------------------------WIKISOURCE HELPER FUNCTIONS-------------------------------------

def _wikisource_html(page_title, lang="en"):
    data = get_json(
        f"https://{lang}.wikisource.org/w/api.php",
        {"action": "parse", "page": page_title, "prop": "text", "format": "json"},
    )
    if "error" in data:
        return None
    return data.get("parse", {}).get("text", {}).get("*", "")


def _strip_noise(soup):
    """Remove scripts, styles and wiki edit/navigation chrome before text extraction."""
    for tag in soup.select("style, script, .mw-editsection, .noprint"):
        tag.decompose()


def _wikisource_page(title, lang="en"):
    """Fetch the text of ONE Wikisource page. Never raises."""
    try:
        html = _wikisource_html(title, lang)
        if html is None:
            return make_result("wikisource", 1, "not_found", title=title)

        soup = BeautifulSoup(html, "html.parser")
        fetched_title = title

        #If Index pages link to the real text on a subpage; follow that link
        target_suffix = "/" + urllib.parse.quote(title.replace(" ", "_"))
        subpage_link = soup.find(
            "a",
            href=lambda h: h
            and h.startswith("/wiki/")
            and h.split("?")[0].endswith(target_suffix),
        )
        if subpage_link:
            subpage_title = urllib.parse.unquote(
                subpage_link["href"].replace("/wiki/", "").replace("_", " ")
            )
            html2 = _wikisource_html(subpage_title, lang)
            if html2:
                soup = BeautifulSoup(html2, "html.parser")
                fetched_title = subpage_title

        _strip_noise(soup)

        url = f"https://{lang}.wikisource.org/wiki/" + urllib.parse.quote(
            fetched_title.replace(" ", "_")
        )

        # Navigation pages (version lists, tables of contents) are mostly links.
        total_chars = len(soup.get_text(" ", strip=True))
        link_chars = sum(len(a.get_text(strip=True)) for a in soup.find_all("a"))
        if total_chars and link_chars / total_chars > 0.4:
            return make_result("wikisource", 1, "index", title=fetched_title, url=url)

        # Wikisource often separates verse lines with <br> inside one paragraph.
        # BeautifulSoup.get_text() can drop those breaks unless they are converted
        # to literal newlines first. Preserve block boundaries too, so stanza and
        # paragraph spacing survives retrieval.
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for block in soup.find_all(
            ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "div"]
        ):
            block.insert_before("\n\n")

        raw_text = soup.get_text(separator="", strip=False).replace("\xa0", " ")
        lines = []
        for raw_line in raw_text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if line:
                lines.append(line)
            elif lines and lines[-1] != "":
                lines.append("")
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        full_text = "\n".join(lines)
        truncated = len(full_text) > MAX_WIKISOURCE_CHARS
        text = full_text[:MAX_WIKISOURCE_CHARS]
        result = make_result(
            "wikisource", 1, "ok", title=fetched_title, url=url, text=text
        )
        result["truncated"] = truncated
        return result

    except Exception as e:
        return make_result("wikisource", 1, "error", title=title, error=str(e))


# -------------------------------------WIKISOURCE TOOL FUNCTIONS-----------------------------------

def search_wikisource_pages(query, lang="en", limit=5):
    """Search one Wikisource language edition without fetching full page text.

    Returns candidate titles, snippets, URLs, and their language so a caller can
    inspect/select a result before fetching it.
    """
    lang = _clean_lang(lang)
    if not (query or "").strip():
        return []

    try:
        data = get_json(
            f"https://{lang}.wikisource.org/w/api.php",
            {
                "action": "query",
                "list": "search",
                "srsearch": query.strip(),
                "srlimit": max(1, min(int(limit), 10)),
                "format": "json",
            },
        )
    except Exception:
        return []

    hits = data.get("query", {}).get("search", [])
    candidates = []
    for hit in hits:
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        candidates.append(
            {
                "title": title,
                "snippet": _strip_html(hit.get("snippet", "")),
                "url": (
                    f"https://{lang}.wikisource.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                ),
                "lang": lang,
            }
        )
    return candidates


def fetch_wikisource_page(title, lang="en"):
    """Fetch one exact Wikisource title, preserving index/subpage handling."""
    return _wikisource_page(title, _clean_lang(lang))


# ===============================TOOL 2: WIKIPEDIA BACKGROUND=================================

def fetch_wikipedia_background(name, lang="en"):
    """Return structured Wikipedia background results without raising errors."""

    lang = _clean_lang(lang)
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

    try:
        search = get_json(
            api_url,
            {
                "action": "query",
                "list": "search",
                "srsearch": name,
                "srlimit": 3,
                "format": "json",
            },
        )

        hits = search.get("query", {}).get("search", [])
        if not hits:
            return [
                make_result("wikipedia", 1, "not_found", title=name)
            ]

        titles = [h["title"] for h in hits]

        page_data = get_json(
            api_url,
            {
                "action": "query",
                "prop": "extracts|pageprops",
                "ppprop": "disambiguation",
                "exintro": 1,
                "explaintext": 1,
                "exsentences": 5,
                "redirects": 1,
                "titles": "|".join(titles),
                "format": "json",
            },
        )

        pages = {
            p.get("title"): p
            for p in page_data.get("query", {}).get("pages", {}).values()
        }

        # Walk the hits in search order; skip disambiguation pages.
        saw_disambiguation = False
        for title in titles:
            page = pages.get(title, {})

            if "disambiguation" in page.get("pageprops", {}):
                saw_disambiguation = True
                continue

            summary = (page.get("extract") or "").strip()
            if not summary:
                continue

            url = (
                f"https://{lang}.wikipedia.org/wiki/"
                + urllib.parse.quote(title.replace(" ", "_"))
            )

            return [
                make_result(
                    "wikipedia",
                    1,
                    "ok",
                    title=title,
                    url=url,
                    text=summary,
                )
            ]

        if saw_disambiguation:
            return [
                make_result(
                    "wikipedia",
                    1,
                    "not_found",
                    title=titles[0],
                    error="Ambiguous name (disambiguation page)",
                )
            ]

        return [
            make_result("wikipedia", 1, "not_found", title=titles[0])
        ]

    except Exception as e:
        return [
            make_result(
                "wikipedia",
                1,
                "error",
                title=name,
                error=str(e),
            )
        ]


# ===============================TOOL 3: TRUSTED LITERARY-SITE SEARCH================================

TRUSTED_LITERARY_DOMAINS = [
    "rekhta.org",
    "hindwi.org",
    "lithub.com",
]

# Search-result snippets are often incomplete. For full-text requests, fetch the
# result page itself and extract its main readable content.
MAX_LITERARY_CHARS = 30000
MAX_SEARCH_CHARS = 30000
MAX_FULL_PAGE_CHARS = 50000


# --------------------------SEARCH HELPER FUNCTIONS (LITERARY SITES + WEB)------------------------------

def _extract_full_page_text(url, max_chars=MAX_FULL_PAGE_CHARS):
    """Fetch a result URL and extract readable page content with line breaks.

    This is a general HTML extraction fallback
    It prefers semantic main/article containers, removes common page chrome, and
    preserves line breaks used by poems wherever the source HTML exposes them.
    Returns (text, truncated). An empty string means extraction was unsuccessful.
    """
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "", False

    try:
        response = requests.get(
            url,
            headers={**HEADERS, "Accept": "text/html,application/xhtml+xml"},
            timeout=15,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type and "xhtml" not in content_type:
            return "", False

        soup = BeautifulSoup(response.content, "html.parser")
        for node in soup.select(
            "script, style, noscript, svg, nav, header, footer, aside, form, "
            "button, iframe, .mw-editsection, .share, .social-share"
        ):
            node.decompose()

        candidates = soup.select(
            "main, article, [role='main'], .entry-content, .post-content, "
            ".article-content, .content"
        )
        if not candidates:
            candidates = [soup.body or soup]

        def content_score(node):
            visible = node.get_text(" ", strip=True)
            if len(visible) < 40:
                return 0
            link_chars = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
            link_ratio = min(1.0, link_chars / max(1, len(visible)))
            return len(visible) * (1.0 - 0.8 * link_ratio)

        content = max(candidates, key=content_score)
        if content_score(content) < 40:
            return "", False

        # Use explicit newlines for common block elements and <br>. Unlike
        # get_text("\n"), this does not split every inline span into a new line.
        for br in content.find_all("br"):
            br.replace_with("\n")
        for tag in content.find_all(
            ["p", "div", "li", "h1", "h2", "h3", "h4", "blockquote", "section", "pre"]
        ):
            tag.insert_before("\n")

        raw_text = content.get_text(separator="", strip=False).replace("\xa0", " ")
        lines = []
        for line in raw_text.splitlines():
            line = " ".join(line.split()).strip()
            if line:
                lines.append(line)
            elif lines and lines[-1] != "":
                # Keep a single blank line between paragraph/stanza blocks.
                lines.append("")
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        text = "\n".join(lines)
        truncated = len(text) > max_chars
        return text[:max_chars], truncated
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Direct page extraction failed for %s: %s", url, exc
        )
        return "", False


def _search_docs(response, source, tier, max_chars, fetch_full_page=False):
    """Normalize Tavily results and optionally replace snippets with page text."""
    docs = []
    for item in response.get("results", []):
        url = item.get("url", "")
        raw_text = item.get("raw_content") or item.get("content") or ""
        text = raw_text[:max_chars]
        truncated = len(raw_text) > max_chars
        extraction = "tavily"

        if fetch_full_page and url:
            page_text, page_truncated = _extract_full_page_text(url)
            # A fetched main-content body is preferable to search snippets even
            # when it is shorter, provided it yielded a meaningful text body.
            if len(page_text) >= 40:
                text = page_text
                truncated = page_truncated
                extraction = "direct_page"

        if not text.strip():
            continue

        doc = make_result(
            source,
            tier,
            "ok",
            title=item.get("title", ""),
            url=url,
            text=text,
        )
        doc["truncated"] = truncated
        doc["text_extraction"] = extraction
        docs.append(doc)

    return docs


# -----------------------------------LITERARY SITES TOOL FUNCTIONS--------------------------------------

def search_literary_sites(query, max_results=3, fetch_full_page=False):
    """Search trusted literary websites; optionally fetch full result pages."""
    try:
        response = _get_tavily().search(
            query=query,
            include_domains=TRUSTED_LITERARY_DOMAINS,
            max_results=max_results,
            include_raw_content=True,
        )
    except Exception as e:
        return [make_result("literary_sites", 1, "error", title=query, error=str(e))]

    docs = _search_docs(
        response, "literary_sites", 1, MAX_LITERARY_CHARS,
        fetch_full_page=fetch_full_page,
    )
    return docs or [make_result("literary_sites", 1, "not_found", title=query)]


# ===============================TOOL 4: OPEN-WEB SEARCH===========================================

def search_web(query, max_results=3, fetch_full_page=False):
    """Search the open web (tier 2, unverified), optionally fetching result pages."""
    try:
        response = _get_tavily().search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
        )
    except Exception as e:
        return [make_result("web_search", 2, "error", title=query, error=str(e))]

    docs = _search_docs(
        response, "web_search", 2, MAX_SEARCH_CHARS,
        fetch_full_page=fetch_full_page,
    )
    return docs or [make_result("web_search", 2, "not_found", title=query)]


# ===============================TOOL 5: OCR FOR UPLOADED IMAGES====================================

def extract_text_from_upload(image_path: str, lang: str = "eng+hin"):
    """
    Extract text from an uploaded image of a poem or literary work.
    Supports English and Hindi by default.
    """

    try:
        # Open the uploaded image.
        img = Image.open(image_path)

        # Extract text using Tesseract OCR.
        text = pytesseract.image_to_string(img, lang=lang).strip()

        # Return the extracted text if any was found.
        if text:
            return [
                make_result(
                    "ocr",
                    1,
                    "ok",
                    title="Extracted text from uploaded image",
                    text=text,
                )
            ]

        # OCR ran successfully but found no readable text.
        return [
            make_result(
                "ocr",
                1,
                "not_found",
                title="Uploaded image",
                error="Unable to extract text from the image.",
            )
        ]

    except Exception as e:
        # Return errors in the same format as our other tools.
        return [
            make_result(
                "ocr",
                1,
                "error",
                title="Uploaded image",
                error=str(e),
            )
        ]


# =----------------------------IMAGE RELEVANCE HELPER FUNCTIONS-----------------------------------

_IMAGE_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "about", "image",
    "photo", "photograph", "picture", "illustration", "scene", "literary",
    "literature", "poem", "poetry", "novel", "book", "work", "author",
    "painting", "portrait", "by", "in", "of", "on", "at", "to", "a", "an",
    "is", "its", "his", "her", "their", "or", "as", "it", "into", "near",
}
# A one-word author name (e.g. "Smith") is accepted only when the image text
# also contains one of these words, so it is clearly a writer and not a random person.
_LITERARY_ROLE_WORDS = {
    "poet", "poetess", "writer", "author", "novelist", "playwright",
    "dramatist", "essayist", "poetry", "poem", "literature", "literary",
    "ghazal", "shayar",
}


def _normal_tokens(value):
    return re.findall(r"[a-z0-9]+", str(value or "").casefold())


def _contains_token_phrase(haystack, phrase):
    text_tokens = _normal_tokens(haystack)
    phrase_tokens = _normal_tokens(phrase)
    width = len(phrase_tokens)
    return bool(width and width <= len(text_tokens) and any(
        text_tokens[index:index + width] == phrase_tokens
        for index in range(len(text_tokens) - width + 1)
    ))


def _image_fields(result):
    fields = [
        result.get("title", ""),
        result.get("object_name", ""),
        result.get("image_description", ""),
        result.get("categories", ""),
        result.get("alt_description", ""),
        result.get("description", ""),
        " ".join(result.get("tags", [])) if isinstance(result.get("tags"), list)
        else result.get("tags", ""),
    ]
    return [str(value) for value in fields if value]


def _author_aliases(name):
    tokens = _normal_tokens(name)
    if not tokens:
        return []
    normalized_name = " ".join(tokens)
    if len(tokens) == 1:
        return [normalized_name]
    aliases = {normalized_name}
    if len(tokens) >= 3 and all(len(token) > 1 for token in tokens[:-1]):
        aliases.add(" ".join([*(token[0] for token in tokens[:-1]), tokens[-1]]))
        aliases.add(" ".join([tokens[0], tokens[-2][0], tokens[-1]]))
    return sorted(aliases, key=len, reverse=True)


def image_mentions_author(result, name):
    """Match the author's name in the image text. A one-word name is accepted
    only if the image text also has a literary role word (poet, writer, ...)."""
    aliases = _author_aliases(name)
    if not aliases:
        return False
    fields = _image_fields(result)
    if len(_normal_tokens(name)) == 1:
        field_tokens = {token for field in fields for token in _normal_tokens(field)}
        if not field_tokens & _LITERARY_ROLE_WORDS:
            return False
    return any(
        _contains_token_phrase(field, alias)
        for field in fields
        for alias in aliases
    )


def _scene_terms(*texts):
    return {
        token for text in texts for token in _normal_tokens(text)
        if len(token) >= 4 and token not in _IMAGE_STOPWORDS
    }


def work_image_relevant(result, work_title, scene_query):
    """Reject candidates that match neither the specific work nor its described scene."""
    fields = _image_fields(result)
    field_tokens = {token for field in fields for token in _normal_tokens(field)}
    field_text = " ".join(fields)
    title_terms = {
        token for token in _normal_tokens(work_title)
        if len(token) > 2 and token not in _IMAGE_STOPWORDS
    }
    distinctive = {token for token in title_terms if len(token) >= 7}
    if distinctive & field_tokens:
        return True
    if title_terms and len(title_terms & field_tokens) >= min(2, len(title_terms)):
        return True
    title_phrase = " ".join(_normal_tokens(work_title))
    if title_phrase and _contains_token_phrase(field_text, title_phrase):
        return True
    scene_terms = _scene_terms(scene_query)
    return bool(scene_terms and len(scene_terms & field_tokens) >= min(2, len(scene_terms)))


def unsplash_scene_relevant(result, scene_query):
    scene_terms = _scene_terms(scene_query)
    field_tokens = {token for field in _image_fields(result) for token in _normal_tokens(field)}
    return bool(scene_terms and len(scene_terms & field_tokens) >= min(2, len(scene_terms)))


# ===============================TOOL 6: UNSPLASH IMAGE SEARCH==================================

UNSPLASH_API = "https://api.unsplash.com/search/photos"


def search_unsplash_images(query, per_page=1):
    """Search Unsplash and return multiple candidates with descriptive metadata.

    Keeping more than one candidate lets the graph reject vague or irrelevant
    photographs instead of automatically displaying the first API result.
    """
    query = (query or "").strip()
    if not query:
        return [make_result("unsplash", 1, "not_found", title=query)]

    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        return [make_result(
            "unsplash", 1, "error", title=query,
            error="UNSPLASH_ACCESS_KEY is not configured.",
        )]

    try:
        response = requests.get(
            UNSPLASH_API,
            params={
                "query": query,
                "per_page": max(1, min(int(per_page), 10)),
                "orientation": "landscape",
            },
            headers={"Authorization": f"Client-ID {access_key}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return [make_result("unsplash", 1, "error", title=query, error=str(e))]

    found = []
    for photo in data.get("results", []):
        urls = photo.get("urls", {})
        photographer = photo.get("user", {})
        image_url = urls.get("regular") or urls.get("small", "")
        if not image_url:
            continue

        description = (photo.get("description") or "").strip()
        alt_description = (photo.get("alt_description") or "").strip()
        tags = [
            (tag.get("title") or "").strip()
            for tag in (photo.get("tags") or [])
            if (tag.get("title") or "").strip()
        ]
        page_url = photo.get("links", {}).get("html", "")
        photographer_url = photographer.get("links", {}).get("html", "")
        download_location = photo.get("links", {}).get("download_location", "")

        result = make_result(
            "unsplash", 1, "ok",
            title=alt_description or description or query,
            url=image_url,
            text=(
                f"Alt description: {alt_description}\n"
                f"Photo description: {description}\n"
                f"Tags: {', '.join(tags)}\n"
                f"Photographer: {photographer.get('name', 'Unknown')}\n"
                f"Photographer profile: {photographer_url}\n"
                f"Image page: {page_url}\n"
                f"Download tracking URL: {download_location}"
            ),
            alt_description=alt_description,
            description=description,
            tags=tags,
            photographer=photographer.get("name", "Unknown"),
            photographer_url=photographer_url,
            page_url=page_url,
            download_location=download_location,
        )
        found.append(result)
        if len(found) >= max(1, min(int(per_page), 10)):
            break

    return found or [make_result("unsplash", 1, "not_found", title=query)]


# ===============================TOOL 7: WIKIMEDIA COMMONS IMAGE SEARCH======================

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}


def search_wikimedia_images(query, per_page=3):
    """Search Wikimedia Commons and return up to per_page usable image results."""
    query = (query or "").strip()
    if not query:
        return [make_result("wikimedia", 1, "not_found", title=query)]

    try:
        data = get_json(
            COMMONS_API,
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                # Commons can return documents/audio among search hits; filter below.
                "gsrlimit": max(10, min(int(per_page), 10)),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime",
                "iiurlwidth": 800,
                "format": "json",
            },
        )
    except Exception as e:
        return [
            make_result(
                "wikimedia", 1, "error", title=query, error=str(e)
            )
        ]

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return [make_result("wikimedia", 1, "not_found", title=query)]

    found = []
    # The pages dict is not in relevance order; "index" is the search rank.
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        image_info = page.get("imageinfo", [])
        if not image_info:
            continue

        info = image_info[0]
        if info.get("mime") not in ALLOWED_IMAGE_MIMES:
            continue

        image_url = info.get("thumburl") or info.get("url", "")
        if not image_url:
            continue

        metadata = info.get("extmetadata", {})
        page_url = info.get("descriptionurl", "")
        artist = _strip_html(
            metadata.get("Artist", {}).get("value", "")
        ) or "Unknown"
        description = _strip_html(
            metadata.get("ImageDescription", {}).get("value", "")
        )
        object_name = _strip_html(
            metadata.get("ObjectName", {}).get("value", "")
        )
        categories = _strip_html(
            metadata.get("Categories", {}).get("value", "")
        )
        license_name = (
            metadata.get("LicenseShortName", {}).get("value", "")
            or "License not specified"
        )
        license_url = metadata.get("LicenseUrl", {}).get("value", "")

        result = make_result(
            "wikimedia",
            1,
            "ok",
            title=page.get("title", ""),
            url=image_url,
            text=(
                f"Image page: {page_url}\n"
                f"Artist: {artist}\n"
                f"Object name: {object_name}\n"
                f"Image description: {description[:1200]}\n"
                f"Categories: {categories[:600]}\n"
                f"License: {license_name}\n"
                f"License URL: {license_url}"
            ),
            page_url=page_url,
            artist=artist,
            object_name=object_name,
            image_description=description,
            categories=categories,
            license=license_name,
            license_url=license_url,
        )
        found.append(result)
        if len(found) >= max(1, int(per_page)):
            break

    return found or [make_result("wikimedia", 1, "not_found", title=query)]