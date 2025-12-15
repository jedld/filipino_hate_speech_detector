#!/usr/bin/env python3
"""Utility to crawl whitelisted domains and build a plain-text corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = (
    "ai251-project-crawler/1.0 (+https://github.com/jedld/filipino_hate_speech_detector)"
)
TEXT_LIKE_MIME = ("text/html", "application/xhtml+xml")
REMOVABLE_TAGS = {"script", "style", "header", "footer", "nav", "noscript", "svg"}
DEFAULT_OUTPUT_FILENAME = "web_corpus.jsonl"


@dataclass
class CrawlTask:
    url: str
    depth: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl a whitelist of domains and export a JSONL text corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-url",
        action="append",
        dest="start_urls",
        help="Seed URL to start crawling (can be provided multiple times).",
    )
    parser.add_argument(
        "--start-url-file",
        type=Path,
        help="Path to a text file that lists seed URLs (one per line).",
    )
    parser.add_argument(
        "--allowed-domain",
        action="append",
        dest="allowed_domains",
        help="Domain allowed for crawling (example: example.com).",
    )
    parser.add_argument(
        "--allowed-domain-file",
        type=Path,
        help="Text file containing allowed domains (one per line).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=250,
        help="Maximum number of pages to keep in the corpus.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum crawl depth from any seed URL.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=200,
        help="Minimum number of characters required to keep a page.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between page fetches to avoid hammering servers.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for main page downloads (set 0 to disable).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header used for HTTP requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "raw" / DEFAULT_OUTPUT_FILENAME,
        help="Output JSONL file path or directory.",
    )
    parser.add_argument(
        "--robots-timeout",
        type=float,
        default=5.0,
        help="Timeout to use specifically when requesting robots.txt files.",
    )
    parser.add_argument(
        "--robots-retries",
        type=int,
        default=0,
        help="Retry attempts when fetching robots.txt (keep low to avoid hangs).",
    )
    parser.add_argument(
        "--no-robots",
        dest="respect_robots",
        action="store_false",
        help="Ignore robots.txt (not recommended).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )
    parser.set_defaults(respect_robots=True)
    return parser


def load_list_from_file(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def normalize_url(url: str) -> str:
    clean_url, _ = urldefrag(url)
    parsed = urlparse(clean_url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    normalized = f"{scheme}://{netloc}{path}"
    if parsed.query:
        normalized = f"{normalized}?{parsed.query}"
    return normalized


def is_url_allowed(url: str, allowed_rules: List[Tuple[str, str]]) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path

    for rule_netloc, rule_path in allowed_rules:
        if netloc == rule_netloc or netloc.endswith(f".{rule_netloc}"):
            if not rule_path or path.startswith(rule_path):
                return True
    return False


def configure_session(
    user_agent: str,
    timeout: float,
    retries: int,
    backoff_factor: float = 0.5,
) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    })
    retries = max(0, retries)
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False, # Don't raise exception on status codes, let requests handle it
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request = _wrap_request_with_timeout(session.request, timeout)
    return session


def _wrap_request_with_timeout(func, timeout: float):
    def wrapped(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return func(method, url, **kwargs)

    return wrapped


class RobotsCache:
    def __init__(
        self,
        session: requests.Session,
        user_agent: str,
        timeout: float,
    ) -> None:
        self.session = session
        self.user_agent = user_agent
        self.timeout = timeout
        self.cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        parser = self._get_parser(parsed.scheme or "https", netloc)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, parsed.path or "/")

    def _get_parser(
        self, scheme: str, netloc: str
    ) -> Optional[robotparser.RobotFileParser]:
        if netloc in self.cache:
            return self.cache[netloc]
        robots_url = f"{scheme}://{netloc}/robots.txt"
        parser = robotparser.RobotFileParser()
        try:
            response = self.session.get(robots_url, timeout=self.timeout)
            if response.ok:
                parser.parse(response.text.splitlines())
                self.cache[netloc] = parser
            else:
                self.cache[netloc] = None
        except requests.RequestException:
            self.cache[netloc] = None
        return self.cache[netloc]


def extract_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(REMOVABLE_TAGS):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n")
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(cleaned_lines)
    return title, cleaned


def enqueue_links(
    soup: BeautifulSoup,
    base_url: str,
    queue: Deque[CrawlTask],
    depth: int,
    max_depth: int,
    allowed_rules: List[Tuple[str, str]],
    known_urls: Set[str],
) -> None:
    if depth >= max_depth:
        return
    for anchor in soup.find_all("a", href=True):
        candidate = urljoin(base_url, anchor["href"])
        normalized = normalize_url(candidate)
        if normalized in known_urls:
            continue
        if not normalized.startswith(("http://", "https://")):
            continue
        if not is_url_allowed(normalized, allowed_rules):
            continue
        queue.append(CrawlTask(normalized, depth + 1))
        known_urls.add(normalized)


def crawl(args: argparse.Namespace) -> None:
    start_urls = set(args.start_urls or [])
    start_urls.update(load_list_from_file(args.start_url_file))
    domain_entries: List[str] = []
    if args.allowed_domains:
        domain_entries.extend(args.allowed_domains)
    domain_entries.extend(load_list_from_file(args.allowed_domain_file))
    
    allowed_rules: List[Tuple[str, str]] = []
    for entry in domain_entries:
        entry = entry.strip().lower()
        # Ensure we can parse it as a URL to extract netloc and path
        parsing_entry = entry
        if not parsing_entry.startswith(("http://", "https://")):
            parsing_entry = f"http://{entry}"
        
        try:
            parsed = urlparse(parsing_entry)
            netloc = parsed.netloc
            path = parsed.path
            if path == "/":
                path = ""
            if netloc:
                allowed_rules.append((netloc, path))
            else:
                # Fallback if parsing fails to find netloc (unlikely with http prefix)
                allowed_rules.append((entry, ""))
        except ValueError:
            logging.warning("Could not parse allowed domain entry: %s", entry)

    start_urls = {normalize_url(url) for url in start_urls}

    if not start_urls:
        raise ValueError("Provide at least one --start-url or --start-url-file entry.")
    if not allowed_rules:
        raise ValueError(
            "Provide at least one --allowed-domain or --allowed-domain-file entry."
        )

    output_path: Path = args.output
    if output_path.exists() and output_path.is_dir():
        logging.info(
            "Output path %s is a directory; writing %s inside it.",
            output_path,
            DEFAULT_OUTPUT_FILENAME,
        )
        output_path = output_path / DEFAULT_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = configure_session(
        args.user_agent,
        args.timeout,
        retries=args.max_retries,
    )
    robots_session = configure_session(
        args.user_agent,
        args.robots_timeout,
        retries=args.robots_retries,
        backoff_factor=0.1,
    )
    robots_cache = RobotsCache(robots_session, args.user_agent, args.robots_timeout)

    visited: Set[str] = set()
    seen_hashes: Set[str] = set()
    known_urls: Set[str] = set(start_urls)
    queue: Deque[CrawlTask] = deque(CrawlTask(url, 0) for url in start_urls)
    saved_pages = 0

    with output_path.open("w", encoding="utf-8") as handle:
        while queue and saved_pages < args.max_pages:
            task = queue.popleft()
            if task.url in visited:
                continue
            visited.add(task.url)

            if not is_url_allowed(task.url, allowed_rules):
                logging.debug("Skipping %s (domain not allowed)", task.url)
                continue
            if args.respect_robots and not robots_cache.can_fetch(task.url):
                logging.info("Blocked by robots.txt: %s", task.url)
                continue

            try:
                response = session.get(task.url)
                response.raise_for_status()
            except requests.RequestException as exc:
                logging.warning("Request failed for %s: %s", task.url, exc)
                if args.verbose and isinstance(exc, requests.HTTPError) and exc.response is not None:
                    logging.debug("Error response content: %s", exc.response.text[:500])
                continue

            content_type = response.headers.get("Content-Type", "").split(";")[0]
            if all(mime not in content_type for mime in TEXT_LIKE_MIME):
                logging.debug("Skipping %s (content-type %s)", task.url, content_type)
                continue

            title, text = extract_text(response.text)
            if len(text) < args.min_text_length:
                logging.debug("Skipping %s (text too short)", task.url)
                continue

            content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                logging.debug("Skipping %s (duplicate content)", task.url)
                continue
            seen_hashes.add(content_hash)

            handle.write(
                json.dumps({"url": task.url, "title": title, "text": text}, ensure_ascii=False)
                + "\n"
            )
            handle.flush()
            saved_pages += 1
            logging.info("Captured %s (chars=%d)", task.url, len(text))

            soup = BeautifulSoup(response.text, "html.parser")
            enqueue_links(
                soup,
                task.url,
                queue,
                task.depth,
                args.max_depth,
                allowed_rules,
                known_urls,
            )

            if args.rate_limit > 0:
                sleep(args.rate_limit)

    if not saved_pages:
        logging.warning("No pages captured. Consider relaxing filters.")
    else:
        logging.info("Saved %d pages to %s", saved_pages, output_path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        crawl(args)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user; partial corpus may be incomplete.")
        return 130
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Scraper failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
