"""PDF text extraction and page rendering with PyMuPDF."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

CAPTION_RE = re.compile(r"^\s*((?:Figure|Fig\.|Table)\s*\d+[a-z]?)\s*[:.|]\s*(.*)", re.IGNORECASE)


@dataclass
class PageInfo:
    number: int  # 1-based
    text: str
    captions: list[str] = field(default_factory=list)
    image_count: int = 0
    drawing_count: int = 0

    @property
    def has_visual(self) -> bool:
        return bool(self.captions) or self.image_count > 0 or self.drawing_count > 40


@dataclass
class ParsedPaper:
    paper_id: str
    filename: str
    pdf_bytes: bytes
    title: str
    pages: list[PageInfo]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def visual_pages(self, limit: int) -> list[PageInfo]:
        """Pages most likely to hold figures or tables, captioned pages first."""
        candidates = [p for p in self.pages if p.has_visual]
        candidates.sort(key=lambda p: (not p.captions, -p.image_count, p.number))
        return sorted(candidates[:limit], key=lambda p: p.number)

    def tagged_text(self, max_chars: int = 60_000) -> str:
        """Full text with [Page N] markers so agents can cite pages."""
        parts, total = [], 0
        for p in self.pages:
            chunk = f"\n[Page {p.number}]\n{p.text.strip()}\n"
            if total + len(chunk) > max_chars:
                parts.append(f"\n[Text truncated after page {p.number - 1}]\n")
                break
            parts.append(chunk)
            total += len(chunk)
        return "".join(parts)


def _normalize_caption(label: str) -> str:
    label = re.sub(r"^fig\.\s*", "Figure ", label.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", label).title()


def parse_pdf(paper_id: str, filename: str, pdf_bytes: bytes) -> ParsedPaper:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages: list[PageInfo] = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        captions = []
        for line in text.splitlines():
            m = CAPTION_RE.match(line)
            if m:
                captions.append(f"{_normalize_caption(m.group(1))}: {m.group(2).strip()}"[:200])
        try:
            drawings = len(page.get_drawings())
        except Exception:
            drawings = 0
        pages.append(
            PageInfo(
                number=i + 1,
                text=text,
                captions=captions,
                image_count=len(page.get_images(full=True)),
                drawing_count=drawings,
            )
        )
    title = (doc.metadata or {}).get("title") or ""
    if not title and pages:
        first_lines = [ln.strip() for ln in pages[0].text.splitlines() if len(ln.strip()) > 8]
        title = first_lines[0] if first_lines else filename
    doc.close()
    return ParsedPaper(paper_id=paper_id, filename=filename, pdf_bytes=pdf_bytes, title=title[:200], pages=pages)


def render_page(pdf_bytes: bytes, page_number: int, dpi: int = 110, fmt: str = "jpeg",
                highlight: list[str] | None = None) -> bytes:
    """Render one page to image bytes, optionally highlighting text snippets."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_number - 1]
        for snippet in highlight or []:
            for rect in find_snippet(page, snippet):
                page.add_highlight_annot(rect)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("jpeg" if fmt == "jpeg" else "png")
    finally:
        doc.close()


def find_snippet(page: "pymupdf.Page", snippet: str) -> list:
    """Locate a quote on a page, falling back to shorter fragments when the model paraphrased slightly."""
    snippet = re.sub(r"\s+", " ", snippet or "").strip().strip('"“”')
    if len(snippet) < 6:
        return []
    for candidate in (snippet, snippet[:80], snippet[:45]):
        hits = page.search_for(candidate)
        if hits:
            return hits
    words = snippet.split()
    for size in (8, 6):
        for start in range(0, max(1, len(words) - size + 1), size):
            hits = page.search_for(" ".join(words[start:start + size]))
            if hits:
                return hits
    return []


def quote_in_page(pdf_bytes: bytes, page_number: int, quote: str) -> bool:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if not 1 <= page_number <= doc.page_count:
            return False
        return bool(find_snippet(doc[page_number - 1], quote))
    finally:
        doc.close()
