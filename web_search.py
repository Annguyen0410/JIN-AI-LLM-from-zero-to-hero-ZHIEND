"""
Lightweight, stdlib-only web search and page fetcher for JinAI.

This is what gives Jin the ability to answer questions about things outside
its training data: the server runs a search, fetches the most relevant
pages, extracts readable text, and feeds a compact "context block" back
into the chat prompt. The model then answers grounded in those snippets.

Design choices
--------------
- **No API keys, no third-party packages.** Search uses DuckDuckGo's
  HTML endpoint which is intentionally easy to scrape. Page fetching is
  plain ``urllib``. A small ``HTMLParser`` subclass turns HTML into
  readable plain text.
- **Polite by default.** Every outgoing request has a User-Agent, a
  short timeout, a hard byte cap, and a global rate limit. We never
  follow ``file://`` or non-HTTP schemes, and we refuse to fetch
  loopback / private IPs (basic SSRF protection).
- **Bounded.** Search returns at most ``max_results`` items, page fetch
  caps response bytes, and the final ``web_context`` block is clipped
  so a tiny SLM can actually fit it in its context window.
"""

from __future__ import annotations

import gzip
import io
import ipaddress
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from html.parser import HTMLParser

# ── Networking knobs ──────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (compatible; JinAI/2.5; +https://github.com/your-jin-ai)"
)
HTTP_TIMEOUT = 8.0
MAX_RESPONSE_BYTES = 2_000_000          # 2 MiB hard cap on any single page
GLOBAL_RATE_LIMIT_RPS = 2.0             # outgoing requests per second
PER_DOMAIN_MIN_INTERVAL = 2.0            # min seconds between requests/domain
SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"

# ── Heuristic: when does a question want a search? ────────────────────
_SEARCH_TRIGGERS = re.compile(
    r"\b("
    r"who|what|when|where|why|how|"
    r"latest|news|today|yesterday|recent|now|currently|"
    r"price|stock|weather|score|result|results|update|"
    r"release|released|launched|announce[sd]?|happened|"
    r"trending|popular|version|deadline|giá|hôm nay|mới nhất|thời tiết|hiện tại|ai|khi nào|ở đâu"
    r")\b",
    re.IGNORECASE,
)


# ── Outgoing rate limiter (very simple sliding window) ────────────────
class _RateGate:
    def __init__(self, rate: float):
        self.interval = 1.0 / max(rate, 0.01)
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            sleep_for = self.next_at - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, 5.0))
            self.next_at = max(now, self.next_at) + self.interval


_GATE = _RateGate(GLOBAL_RATE_LIMIT_RPS)
_DOMAIN_GATE: dict[str, _RateGate] = {}
_DOMAIN_GATE_LOCK = threading.Lock()
_ROBOTS_CACHE: dict[str, object] = {}


def _domain_rate_gate(domain: str) -> _RateGate:
    """Per-domain rate limiter (<=1 req / PER_DOMAIN_MIN_INTERVAL s)."""
    with _DOMAIN_GATE_LOCK:
        gate = _DOMAIN_GATE.get(domain)
        if gate is None:
            gate = _RateGate(1.0 / PER_DOMAIN_MIN_INTERVAL)
            _DOMAIN_GATE[domain] = gate
        return gate


def robots_allows(domain: str, path: str, user_agent: str = "JinAI") -> bool:
    """Respect robots.txt for ``domain`` (cached per domain).

    Uses ``urllib.robotparser``. On any error (no robots, network fail,
    parse fail) we default to ALLOWED for read-only fetches but still
    refuse obvious disallow-all sites.
    """
    import urllib.robotparser
    rp = _ROBOTS_CACHE.get(domain)
    if rp is None:
        rp = urllib.robotparser.RobotFileParser()
        url = f"https://{domain}/robots.txt"
        rp.set_url(url)
        try:
            if not _is_safe_url(url):
                return False
            req = urllib.request.Request(url, headers={"User-Agent":USER_AGENT})
            with urllib.request.build_opener(_SafeRedirect()).open(req, timeout=HTTP_TIMEOUT) as response:
                raw = response.read(128000)
            rp.parse(raw.decode("utf-8", "replace").splitlines())
        except Exception:
            rp = None
        _ROBOTS_CACHE[domain] = rp
    if rp is None:
        return True
    return rp.can_fetch(user_agent, f"https://{domain}{path}")


def _domain_of_url(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


# ── URL normalisation ─────────────────────────────────────────────────
def _ascii_url(url: str) -> str:
    """Percent-encode non-ASCII characters so request lines stay ASCII.

    Search results legitimately point at IRIs (e.g. Vietnamese Wikipedia
    paths such as ``/wiki/Thủ_đô_Hà_Nội``). ``http.client`` encodes the
    request line as ASCII and raises ``UnicodeEncodeError`` on those, which
    surfaced as a 400 error on any Vietnamese question that triggered a
    web lookup. Existing percent-escapes are preserved (no double-encoding).
    """

    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return url
    netloc = parts.netloc
    if netloc and not netloc.isascii():
        try:
            netloc = netloc.encode("idna").decode("ascii")
        except UnicodeError:
            netloc = urllib.parse.quote(netloc, safe=":@[]")
    path = urllib.parse.quote(parts.path, safe="/%:@!$&'()*+,;=~")
    query = urllib.parse.quote(parts.query, safe="=&%:@!$'()*+,;/?~")
    fragment = urllib.parse.quote(parts.fragment, safe="=&%:@!$'()*+,;/?~")
    return urllib.parse.urlunsplit((parts.scheme, netloc, path, query, fragment))


# ── SSRF guard ────────────────────────────────────────────────────────
def _is_safe_url(url: str) -> bool:
    """Reject anything that is not a regular http/https public URL."""

    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if parsed.username or parsed.password:
        return False
    try:
        if parsed.port not in (None, 80, 443):
            return False
    except ValueError:
        return False
    if not host:
        return False
    try:
        if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
            return False
        ip = ipaddress.ip_address(host)
        return ip.is_global and not ip.is_multicast and not ip.is_reserved and not ip.is_unspecified
    except ValueError:
        # Hostname, not an IP literal. Try to resolve.
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        for info in infos:
            sockaddr = info[4]
            try:
                ip = ipaddress.ip_address(sockaddr[0])
            except (ValueError, IndexError):
                continue
            if not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                return False
        return bool(infos)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Redirect targets may be IRIs too; normalise before both the
        # safety check and the (ASCII-only) request line.
        newurl = _ascii_url(newurl)
        if not _is_safe_url(newurl):
            raise RuntimeError("Refused unsafe redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# ── Low-level HTTP ────────────────────────────────────────────────────
def _http_get(url: str, *, params: dict | None = None) -> tuple[str, str]:
    """Return (text, final_url). Raises ``RuntimeError`` on failure."""

    url = _ascii_url(url)
    if params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urllib.parse.urlencode(params)
    if not _is_safe_url(url):
        raise RuntimeError(f"Refused to fetch unsafe URL: {url}")

    _GATE.wait()
    domain = _domain_of_url(url)
    if domain:
        _domain_rate_gate(domain).wait()
        path = urllib.parse.urlparse(url).path or "/"
        if not robots_allows(domain, path):
            raise RuntimeError(f"robots.txt disallows {url}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip",
        },
    )
    try:
        with urllib.request.build_opener(_SafeRedirect()).open(req, timeout=HTTP_TIMEOUT) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if not any(t in ctype.lower() for t in ("text/", "application/xhtml", "application/xml", "application/json")):
                raise RuntimeError(f"Unsupported content-type: {ctype!r}")

            raw = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raw = raw[:MAX_RESPONSE_BYTES]

            if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
                    raw = compressed.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("Decompressed response exceeds byte budget")

            enc = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype, re.IGNORECASE)
            if m:
                enc = m.group(1)
            try:
                text = raw.decode(enc, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")
            return text, resp.geturl()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise RuntimeError(f"HTTP error fetching {url}: {exc}") from exc


# ── HTML → text ───────────────────────────────────────────────────────
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "table",
    "header", "footer", "main", "nav", "aside", "blockquote", "pre",
}
_DROP_TAGS = {"script", "style", "noscript", "head", "svg", "form", "iframe"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_TAGS:
            self.skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _DROP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        self.parts.append(data)


def html_to_text(html_text: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        # The stdlib parser is forgiving but a malformed page can still
        # raise. Fall back to a regex strip.
        return re.sub(r"<[^>]+>", " ", html_text)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── DuckDuckGo search ─────────────────────────────────────────────────
_RESULT_BLOCK_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'(?:.*?<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
    re.IGNORECASE | re.DOTALL,
)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _clean_inline(text: str) -> str:
    text = unescape(_TAG_STRIP_RE.sub("", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _resolve_ddg_url(href: str) -> str:
    """DDG wraps result links as ``//duckduckgo.com/l/?uddg=ENCODED``.
    Unwrap to the real destination so we can fetch it directly."""

    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        qs = urllib.parse.parse_qs(parsed.query)
        target = qs.get("uddg", [None])[0]
        if target:
            return urllib.parse.unquote(target)
    return href


def search(query: str, max_results: int = 5, region: str = "wt-wt") -> list[dict]:
    """Run a DuckDuckGo search and return up to ``max_results`` hits.

    Each hit is a dict with ``title``, ``url`` and ``snippet``.
    """

    query = (query or "").strip()
    if not query:
        return []
    text, _ = _http_get(SEARCH_ENDPOINT, params={"q": query, "kl": region})
    out: list[dict] = []
    seen: set[str] = set()
    for match in _RESULT_BLOCK_RE.finditer(text):
        href = _resolve_ddg_url(match.group(1))
        title = _clean_inline(match.group(2))
        snippet = _clean_inline(match.group(3) or "")
        if not href or not title:
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append({"title": title, "url": href, "snippet": snippet})
        if len(out) >= max_results:
            break
    return out


# ── End-to-end: search + fetch + format ───────────────────────────────
def fetch_text(url: str, *, max_chars: int = 4_000) -> str:
    """Fetch ``url`` and return cleaned text, capped at ``max_chars``."""

    raw, _ = _http_get(url)
    text = html_to_text(raw)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


# Self-referential questions never want the web: searching them drags in
# unrelated pages, and the grounding pipeline then quotes those pages as if
# they were the answer — that is how "can you introduce yourself?" ended up
# citing an article about Vietnamese greetings.
_SELF_REFERENTIAL = re.compile(
    r"who are you|what are you|your name|introduce yourself|about yourself|"
    r"what can you do|how are you\b|"
    r"bạn là ai|mày là ai|bạn tên gì|tên bạn|giới thiệu (?:về )?(?:bạn|bản thân|mình)|"
    r"bạn làm được gì|làm được gì|bạn khỏe không|khỏe không",
    re.IGNORECASE,
)

# Pure small talk. Only skipped when no search trigger is present, so
# "chào bạn, giá vàng hôm nay?" still searches.
_SMALL_TALK = re.compile(
    r"(?:^|[\s,!.?])"
    r"(?:hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay|"
    r"chào|xin chào|chao|cảm ơn|cám ơn|tạm biệt)"
    r"(?:$|[\s,!.?…])",
    re.IGNORECASE,
)


def looks_like_search_query(message: str) -> bool:
    """Heuristic: does this user message want fresh information?

    Chit-chat is filtered *before* the question-mark rule: a question mark is
    not evidence that a web lookup is wanted ("bạn là ai?" is not research).
    """

    if not message:
        return False
    if _SELF_REFERENTIAL.search(message):
        return False
    has_trigger = bool(_SEARCH_TRIGGERS.search(message))
    if _SMALL_TALK.search(message) and not has_trigger:
        return False
    if "?" in message:
        return True
    return has_trigger


def web_context(
    query: str,
    *,
    max_results: int = 4,
    fetch_top: int = 1,
    snippet_chars: int = 240,
    page_chars: int = 1_500,
) -> tuple[str, list[dict]]:
    """Build a compact context block from live web search.

    Returns ``(context_block, sources)``. ``sources`` is a list of
    ``{title, url, snippet}`` records suitable for showing in the UI.
    The context block is plain text, ready to inject into the chat
    prompt before the user's message.
    """

    try:
        results = search(query, max_results=max_results)
    except RuntimeError as exc:
        return f"[web search failed: {exc}]", []

    if not results:
        return "[web search returned no results]", []

    # Optionally fetch the top N pages so the model has real text, not
    # just snippets. This dramatically improves grounded answers.
    enriched: list[dict] = []
    for i, hit in enumerate(results):
        record = dict(hit)
        record["snippet"] = (record.get("snippet") or "")[:snippet_chars]
        if i < fetch_top:
            try:
                record["body"] = fetch_text(hit["url"], max_chars=page_chars)
            except RuntimeError:
                record["body"] = ""
        enriched.append(record)

    lines = ["Web search results for: " + query.strip(), ""]
    for i, rec in enumerate(enriched, 1):
        lines.append(f"[{i}] {rec['title']}")
        lines.append(f"    URL: {rec['url']}")
        if rec.get("snippet"):
            lines.append(f"    Snippet: {rec['snippet']}")
        if rec.get("body"):
            body = rec["body"].replace("\n", " ")
            if len(body) > page_chars:
                body = body[:page_chars] + "…"
            lines.append(f"    Excerpt: {body}")
        lines.append("")
    block = "\n".join(lines).strip()
    return block, enriched


# ── Phase 5: harvest sources (Wikipedia API, arXiv API) ───────────────

def _strip_markup(text: str) -> str:
    """Remove wiki markup / XML tags from API text bodies."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\[\[[^\]]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[[^\]]*\]\]", "", text)
    text = re.sub(r"'''?|''?", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wikipedia_search(query: str, max_results: int = 3) -> list[dict]:
    """Wikipedia API search: returns [{title, url, snippet, body}]."""
    try:
        text, _ = _http_get(
            WIKIPEDIA_API,
            params={
                "action": "query", "format": "json", "list": "search",
                "srsearch": query, "srlimit": max_results, "utf8": "1",
            },
        )
        data = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"Wikipedia search failed: {exc}") from exc
    out = []
    for hit in data.get("query", {}).get("search", [])[:max_results]:
        title = hit.get("title", "")
        pageid = hit.get("pageid")
        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        body = _strip_markup(hit.get("snippet", ""))
        out.append({"title": title, "url": url, "snippet": body, "body": body,
                    "pageid": pageid})
    return out


def wikipedia_extract(title: str, max_chars: int = 3_000) -> str:
    """Fetch the plain-text extract of a Wikipedia article."""
    try:
        text, _ = _http_get(
            WIKIPEDIA_API,
            params={
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": "1", "titles": title, "exintro": "0",
                "exlimit": "1", "utf8": "1",
            },
        )
        data = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"Wikipedia extract failed: {exc}") from exc
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        body = page.get("extract", "") or ""
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + "…"
        return body
    return ""


def arxiv_search(query: str, max_results: int = 3) -> list[dict]:
    """arXiv API (Atom XML): returns [{title, url, snippet, body}]."""
    try:
        text, _ = _http_get(
            ARXIV_API,
            params={"search_query": f"all:{query}", "max_results": max_results},
        )
    except Exception as exc:
        raise RuntimeError(f"arXiv search failed: {exc}") from exc
    import xml.etree.ElementTree as ET
    out = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns)[:max_results]:
        title = _clean_inline(entry.findtext("a:title", "", ns))
        summary = _clean_inline(entry.findtext("a:summary", "", ns))
        url = entry.findtext("a:id", "", ns).strip()
        out.append({"title": title, "url": url, "snippet": summary[:240],
                    "body": summary})
    return out


def harvest(query: str, *, max_results: int = 4, max_chars: int = 2_500) -> list[dict]:
    """Phase 5 harvest facade: combine DuckDuckGo + Wikipedia + arXiv.

    Returns a list of dicts compatible with ``curiosity_engine.HarvestedPage``
    (``url``, ``domain``, ``title``, ``text``, ``kind``). Each source is
    tried independently; a failing source never aborts the others.
    """
    pages: list[dict] = []
    seen: set[str] = set()

    def push(rec: dict, kind: str) -> None:
        url = rec.get("url", "")
        if not url or url in seen:
            return
        seen.add(url)
        body = (rec.get("body") or "").strip()
        if len(body) < 100:
            return
        pages.append({
            "url": url,
            "domain": _domain_of_url(url),
            "title": rec.get("title", ""),
            "text": body[:max_chars],
            "kind": kind,
        })

    # DuckDuckGo + fetch top results.
    try:
        for hit in search(query, max_results=max_results):
            try:
                hit["body"] = fetch_text(hit["url"], max_chars=max_chars)
            except RuntimeError:
                hit["body"] = ""
            push(hit, "web")
    except RuntimeError:
        pass

    # Wikipedia API.
    try:
        for hit in wikipedia_search(query, max_results=2):
            try:
                hit["body"] = wikipedia_extract(hit.get("title", ""),
                                                 max_chars=max_chars)
            except RuntimeError:
                pass
            push(hit, "wikipedia")
    except RuntimeError:
        pass

    # arXiv API (scientific angle).
    try:
        for hit in arxiv_search(query, max_results=2):
            push(hit, "arxiv")
    except RuntimeError:
        pass

    return pages


# ── CLI smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run a quick web search.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--results", type=int, default=4)
    parser.add_argument("--fetch-top", type=int, default=1)
    args = parser.parse_args()

    block, srcs = web_context(
        " ".join(args.query),
        max_results=args.results,
        fetch_top=args.fetch_top,
    )
    enc = (sys.stdout.encoding or "utf-8")
    safe = block.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe)
    print()
    print(f"=== {len(srcs)} sources ===")
    for s in srcs:
        line = f"- {s['title']} -> {s['url']}"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"))
