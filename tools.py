"""
Custom LangChain tools used by the Literature Discussion Agent.
"""

import urllib.parse

import pytesseract
import requests
import wikipedia

from bs4 import BeautifulSoup
from PIL import Image

from tavily import TavilyClient

from langchain_core.tools import tool

from configure import (
    TAVILY_API_KEY,
    UNSPLASH_ACCESS_KEY,
)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"D:\tesseraact-ocr\tesseract.exe"
)

import time
import functools

def timed(func):
    """Wraps a tool function to print how long it actually took — for diagnosing latency."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMING] {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def search_wikisource(query: str, num_results: int = 3) -> str:
    """
    Search Wikisource for literary works and return matching titles,
    snippets, and page URLs.
    """

    url = "https://en.wikisource.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": num_results
    }

    headers = {
        "User-Agent": "LiteratureDiscussionAgent/1.0 (Educational Project)"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

    except requests.exceptions.RequestException as e:
        return f"Request failed:\n{e}"
    except ValueError:
        return f"Failed to decode JSON.\nResponse:\n{response.text}"

    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return "No results found on Wikisource."

    results = []

    for item in search_results:
        title = item["title"]

        snippet = (
            item["snippet"]
            .replace('<span class="searchmatch">', '')
            .replace("</span>", "")
        )

        page_url = (
            "https://en.wikisource.org/wiki/"
            + title.replace(" ", "_")
        )

        results.append(
            f"Title: {title}\n"
            f"Snippet: {snippet}\n"
            f"URL: {page_url}\n"
        )

    return "\n".join(results)

@tool
def search_wikisource_tool(query: str) -> str:
    """
    Search Wikisource for a literary work by title, author, or line.
     Returns search results with snippets and page titles.
      Use get_wikisource_page_content_tool afterward with the exact title to fetch full text.
      """
    return search_wikisource(query)

def get_wikisource_page_content(title: str) -> str:
    def fetch_html(page_title):
        url = "https://en.wikisource.org/w/api.php"
        params = {"action": "parse",
                  "page": page_title,
                  "prop": "text",
                  "format": "json"
                  }

        headers = {
            "User-Agent": "LiteratureDiscussionAgent/1.0 (Educational Project)"
            }

        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
            )

        data = resp.json()

        if "error" in data:
            return None
        return data.get("parse", {}).get("text", {}).get("*", "")

    def light_clean(raw_text):
        # strip empty lines and collapse whitespace
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        return "\n".join(lines)

    html = fetch_html(title)
    if html is None:
        return f"Page '{title}' not found."

    soup = BeautifulSoup(html, "html.parser")

    target_suffix = "/" + urllib.parse.quote(title.replace(" ", "_"))
    subpage_link = soup.find(
        "a",
        href=lambda h: h and h.startswith("/wiki/") and h.split("?")[0].endswith(target_suffix)
        )

    if subpage_link:
        subpage_title = urllib.parse.unquote(subpage_link["href"].replace("/wiki/", "").replace("_", " "))
        html2 = fetch_html(subpage_title)
        if html2:
            soup2 = BeautifulSoup(html2, "html.parser")
            text2 = light_clean(soup2.get_text(separator="\n"))
            return f"(Fetched from: {subpage_title})\n\n{text2[:10000]}"

    text = light_clean(soup.get_text(separator="\n"))
    return f"(Fetched from: {title})\n\n{text[:10000]}"

@tool
def get_wikisource_page_content_tool(title: str) -> str:
    """
    Fetch the full text of a specific Wikisource page,
     given its exact title (get this from search_wikisource_tool first).
      Automatically follows disambiguation/index pages to the real content.
       Raw output may include site navigation, catalog IDs, and copyright notices
        mixed in with the actual text — identify and use only the genuine literary content.
        """
    return get_wikisource_page_content(title)


def get_author_info(name: str, lang: str = "en") -> str:
    """
    Fetch biography/background about an author or literary work from Wikipedia,
    trying the specified language edition (e.g., 'en', 'hi').
    """
    wikipedia.set_lang(lang)
    try:
        summary = wikipedia.summary(
            name,
            sentences=5,
            auto_suggest=True
        )

        page = wikipedia.page(
            name,
            auto_suggest=True
        )

        return f"Language: {lang}\nBiography:\n{summary}\n\nSource URL: {page.url}"

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple matches found, please be more specific: {e.options[:5]}"

    except wikipedia.exceptions.PageError:
        return (f"No Wikipedia page found for '{name}' in language '{lang}', try"
                f" different language or check the spelling.")

    except Exception as e:
        return f"Error fetching info: {e}"

@tool
def get_author_info_tool(name: str, lang: str = "en") -> str:
    """
    Get biography and background info about an author/poet from Wikipedia.
     Pass lang='hi' for Hindi,lang='en' for English/Western authors,
      based on the subject's likely language/origin.
      """
    return get_author_info(name, lang)

def extract_text_from_upload(image_path: str, lang: str = "eng+hin") -> str:
    """
    Extract text from an uploaded image of a poem/literary work using OCR.
    lang: tesseract language code(s), e.g., 'eng', 'hin', or 'eng+hin' for mixed.
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip() if text.strip() else "Unable to extract text from the image."

    except Exception as e:
        return f"OCR failed: {e}"


@tool
def extract_text_from_upload_tool(image_path: str, lang: str = "eng+hin") -> str:
    """
    Extract text from an uploaded image of a poem or literary work using OCR.
     lang can be 'eng', 'hin', or 'eng+hin' for mixed-script images.
     """
    return extract_text_from_upload(image_path, lang)


def search_thematic_image(theme_query: str) -> str:
    """
    Search Unsplash for thematic images.

    Returns the URLs and photographer credits
    for up to three matching images.
    """

    url = "https://api.unsplash.com/search/photos"

    params = {
        "query": theme_query,
        "per_page": 3,
    }

    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()

    except Exception as e:
        return f"Image search failed: {e}"

    results = data.get("results", [])

    if not results:
        return f"No image found for theme: {theme_query}"

    output = []

    for i, photo in enumerate(results, start=1):

        output.append(
            f"Image URL {i}: {photo['urls']['regular']}\n"
            f"Photographer {i}: {photo['user']['name']}"
        )

    return "\n\n".join(output)

@tool
def search_thematic_image_tool(theme_query: str) -> str:
    """
    Search Unsplash for a royalty-free image matching a literary theme
     (e.g., 'war soldier', 'autumn orchard'). First reason about what
      visual theme best represents the work being discussed, then call
       this with those search terms.
       """
    return search_thematic_image(theme_query)


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Trusted literary domains (Tier 1 — no dedicated API, so accessed via domain-restricted search)
TRUSTED_LITERARY_DOMAINS = ["rekhta.org", "hindwi.org", "lithub.com"]

def search_web_tiered(query: str, tier: str = "literary") -> str:
    """
    Search the web for literary content, tiered by reliability.
    tier='literary' -> restricted to trusted literary sites (Rekhta, Hindwi, Literary Hub) — treated as grounded/factual.
    tier='opinions' -> open web search for reviews/discussions/opinions (Reddit, blogs, forums) — treated as subjective, opt-in only.
    """
    try:
        if tier == "literary":
            response = tavily_client.search(
                query=query,
                include_domains=TRUSTED_LITERARY_DOMAINS,
                max_results=3
            )
            label = "TIER 1 — Trusted literary source"
        else:
            response = tavily_client.search(
                query=query,
                max_results=3
            )
            label = "TIER 2 — Unverified/community opinion, NOT confirmed fact"

        results = response.get("results", [])
        if not results:
            return f"No results found ({tier} search)."

        formatted = []
        for r in results:
            formatted.append(
                f"[{label}]\nTitle:"
                f" {r['title']}\nURL:"
                f" {r['url']}\nSnippet:"
                f" {r['content'][:300]}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"Web search failed: {e}"

@tool
def search_web_tiered_tool(query: str, tier: str = "literary") -> str:
    """
    Search the web for literary content. tier='literary' searches trusted
     literary sites (Rekhta, Hindwi, Literary Hub) for grounded factual info
      — use this when Wikisource/Wikipedia don't have something. tier='opinions'
      searches the open web (Reddit, blogs) for reader opinions/reviews —
      ONLY use this if the user has explicitly asked for opinions/reviews/reactions,
       and always label this content as unverified/subjective when presenting it.
       """
    return search_web_tiered(query, tier)


def recommend_similar_works(author_or_theme: str, based_on: str = "author") -> str:
    """
    Suggest 2-3 related literary works by reusing existing search tools.
    based_on='author' -> other notable works by the same author.
    based_on='theme'  -> other works with a similar theme/movement.
    """
    if based_on.lower() == "author":
        query = f"other famous poems or works by {author_or_theme}"
    else:
        query = f"famous poems or literary works about {author_or_theme}"

    # Try Wikisource first (grounded, full-text-capable source)
    wikisource_results = search_wikisource(query, num_results=3)

    # Fall back / supplement with trusted literary web search
    web_results = search_web_tiered(query, tier="literary")

    combined = (f"--- Wikisource suggestions"
                f" ---\n{wikisource_results}\n\n"
                f"--- Trusted literary site suggestions"
                f" ---\n{web_results}")
    return combined

@tool
def recommend_similar_works_tool(author_or_theme: str, based_on: str = "author") -> str:
    """
    Suggest 2-3 related literary works, sourced from Wikisource and trusted
     literary sites. based_on='author' finds other works by the same author;
      based_on='theme' finds other works with a similar theme. Use this at the
       end of a discussion about a specific work or author to offer suggestions.
       """
    return recommend_similar_works(author_or_theme, based_on)


search_wikisource = timed(search_wikisource)
get_wikisource_page_content = timed(get_wikisource_page_content)
get_author_info = timed(get_author_info)
search_thematic_image = timed(search_thematic_image)
search_web_tiered = timed(search_web_tiered)