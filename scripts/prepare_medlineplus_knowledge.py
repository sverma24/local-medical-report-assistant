from __future__ import annotations

import argparse
import html
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "https://medlineplus.gov/lab-tests/"
USER_AGENT = "vaultmd-medlineplus-ingestor/1.0"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in self._skip_tags:
            self._skip_depth += 1
        if self._skip_depth > 0:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        if self._skip_depth > 0:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "section", "article"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        text = " ".join(self._parts)
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return html.unescape(text)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _extract_test_urls(index_html: str) -> list[str]:
    hrefs = re.findall(r'href="([^"]+)"', index_html, flags=re.IGNORECASE)
    seen: set[str] = set()
    urls: list[str] = []
    for href in hrefs:
        full = urllib.parse.urljoin(BASE_URL, href)
        parsed = urllib.parse.urlparse(full)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc != "medlineplus.gov":
            continue
        if not parsed.path.startswith("/lab-tests/"):
            continue
        if parsed.path in {"/lab-tests/", "/lab-tests"}:
            continue
        slug = parsed.path.removeprefix("/lab-tests/").strip("/")
        if " " in slug:
            continue
        if slug.startswith("http"):
            continue
        if not slug:
            continue
        if not re.fullmatch(r"[a-z0-9-]+", slug):
            continue
        if slug.startswith("about") or slug.startswith("lab-tests-a-z"):
            continue
        normalized = f"https://medlineplus.gov/lab-tests/{slug}/"
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
    return sorted(urls)


def _extract_title(page_html: str) -> str:
    title_match = re.search(
        r"<title>(.*?)</title>", page_html, flags=re.IGNORECASE | re.DOTALL
    )
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()
        title = title.replace(" | MedlinePlus", "")
        if title:
            return html.unescape(title)
    h1_match = re.search(
        r"<h1[^>]*>(.*?)</h1>", page_html, flags=re.IGNORECASE | re.DOTALL
    )
    if h1_match:
        h1 = re.sub(r"<[^>]+>", " ", h1_match.group(1))
        h1 = re.sub(r"\s+", " ", h1).strip()
        if h1:
            return html.unescape(h1)
    return "MedlinePlus Lab Test"


def _slug_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    slug = path.removeprefix("/lab-tests/").strip("/")
    return slug or "unknown-test"


def _write_markdown(url: str, page_html: str, output_dir: Path) -> Path:
    slug = _slug_from_url(url)
    title = _extract_title(page_html)
    parser = _TextExtractor()
    parser.feed(page_html)
    body = parser.get_text()

    lines = [
        f"# {title}",
        "",
        f"Source: {url}",
        "",
        body,
        "",
    ]
    content = "\n".join(lines)
    target = output_dir / f"{slug}.md"
    target.write_text(content, encoding="utf-8")
    return target


def prepare_medlineplus_knowledge(output_dir: Path, clean: bool) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    index_html = _fetch(BASE_URL)
    test_urls = _extract_test_urls(index_html)
    written = 0
    failed = 0

    for url in test_urls:
        try:
            page_html = _fetch(url)
            _write_markdown(url, page_html, output_dir)
            written += 1
        except (urllib.error.URLError, TimeoutError, ValueError):
            failed += 1

    return written, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all MedlinePlus lab tests into markdown files."
    )
    parser.add_argument(
        "--output-dir",
        default="data/knowledge/medlineplus",
        help="Directory to write markdown files into.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output directory before ingesting.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    written, failed = prepare_medlineplus_knowledge(output_dir=output_dir, clean=args.clean)
    print(f"medlineplus_pages_written={written}")
    print(f"medlineplus_pages_failed={failed}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
