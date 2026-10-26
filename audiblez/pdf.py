# -*- coding: utf-8 -*-
"""Text extraction from (text-based) PDFs, rebuilt into paragraphs and split into chapters.

PDFs have no notion of paragraphs, only positioned lines, so this module:
  1. reads the PDF one page at a time (pdfminer), keeping only light-weight line records,
  2. drops running headers/footers, page numbers, footnote-sized text and off-page text,
  3. rebuilds paragraphs from first-line indents, vertical gaps and short last lines, joining
     paragraphs that continue across page breaks and undoing end-of-line hyphenation,
  4. splits the paragraphs into chapters using the PDF outline (bookmarks), or larger-font headings,
     and caps every chapter at MAX_SECTION_CHARS, always cutting between paragraphs.
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from types import SimpleNamespace

MAX_SECTION_CHARS = 40_000  # ~40 minutes of audio; long sections are split between paragraphs
MARGIN_ZONE = 0.12  # top/bottom fraction of the page where running headers and footers live
SKIP_TITLES = re.compile(r'^(cover|title page|contents|table of contents|index|copyright|about this (book|e-?book)|'
                         r'indexes|index of .*|colophon)$', re.I)


@dataclass
class Line:
    text: str
    page: int
    x0: float
    x1: float
    top: float  # distance from the top of the page
    bottom: float
    size: float  # median font size
    zone: bool = False  # in the header/footer zone

    @property
    def height(self):
        return self.bottom - self.top


@dataclass
class Paragraph:
    text: str
    page: int
    heading: bool = False


@dataclass
class PdfChapter:
    title: str
    paragraphs: list = field(default_factory=list)
    chapter_index: int = 0
    is_selected: bool = True

    @property
    def extracted_text(self):
        return '\n'.join(p.text for p in self.paragraphs)

    def get_name(self):
        return self.title


def load_pdf(file_path, max_section_chars=MAX_SECTION_CHARS):
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfpage import PDFPage

    with open(file_path, 'rb') as f:
        doc = PDFDocument(PDFParser(f))
        info = doc.info[0] if doc.info else {}
        title, creator = pdf_info_text(info.get('Title')), pdf_info_text(info.get('Author'))
        page_index = {page.pageid: i for i, page in enumerate(PDFPage.create_pages(doc))}
        outline = read_outline(doc, page_index)

    lines = extract_lines(file_path, num_pages=len(page_index))
    lines = remove_headers_and_footers(lines, len(page_index))
    body_size = median_font_size(lines)
    paragraphs = build_paragraphs(lines, body_size)
    chapters = split_into_chapters(paragraphs, outline, max_section_chars)
    for i, c in enumerate(chapters):
        c.chapter_index = i
    good = [c for c in chapters if len(c.extracted_text) > 200 and not SKIP_TITLES.match(c.title.strip())]
    print(f'PDF: {len(page_index)} pages, {len(paragraphs)} paragraphs, {len(chapters)} sections'
          f' ({"from the outline" if outline else "from headings"})')
    return SimpleNamespace(title=title or '', creator=creator or '', cover_image=b'', chapters=chapters,
                           good_chapters=good or chapters)


def pdf_info_text(value):
    from pdfminer.pdftypes import resolve1
    from pdfminer.utils import decode_text
    value = resolve1(value)
    if isinstance(value, bytes):
        return decode_text(value).strip()
    return str(value).strip() if value else ''


def read_outline(doc, page_index):
    """Returns [(level, title, page_number)] for outline entries that point at a page, in page order."""
    from pdfminer.pdfdocument import PDFNoOutlines
    from pdfminer.pdftypes import resolve1, PDFObjRef
    entries = []
    try:
        for level, title, dest, action, _ in doc.get_outlines():
            if dest is None and action:
                action = resolve1(action)
                dest = action.get('D') if isinstance(action, dict) else None
            try:
                dest = resolve1(dest)
                if isinstance(dest, (str, bytes)):
                    dest = resolve1(doc.get_dest(dest))
                if isinstance(dest, dict):
                    dest = resolve1(dest.get('D'))
                page_ref = dest[0] if isinstance(dest, list) and dest else None
                page = page_index.get(page_ref.objid) if isinstance(page_ref, PDFObjRef) else None
            except Exception:
                page = None
            if page is not None and title:
                entries.append((level, title.strip(), page))
    except PDFNoOutlines:
        return []
    except Exception as e:
        print(f'Could not read the PDF outline ({e}), detecting chapters from headings instead.')
        return []
    if not entries:
        return []
    # Use the shallowest level that actually splits the book (a single top-level "Book" entry doesn't)
    for level in sorted({e[0] for e in entries}):
        at_level = [e for e in entries if e[0] == level]
        if len({e[2] for e in at_level}) >= 2:
            return sorted(at_level, key=lambda e: e[2])
    return []


def extract_lines(file_path, num_pages):
    """Reads the PDF page by page, returning only Line records so memory stays small for big PDFs."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTChar, LTTextContainer, LTTextLineHorizontal

    def iter_lines(container):
        for el in container:
            if isinstance(el, LTTextLineHorizontal):
                yield el
            elif isinstance(el, LTTextContainer):
                yield from iter_lines(el)

    lines = []
    for page_no, page in enumerate(extract_pages(file_path, laparams=LAParams())):
        if page_no and page_no % 50 == 0:
            print(f'Reading PDF page {page_no}/{num_pages}...')
        for el in page:
            if not isinstance(el, LTTextContainer):
                continue
            for line in iter_lines([el]):
                if line.x1 < 0 or line.x0 > page.width or line.y1 < 0 or line.y0 > page.height:
                    continue  # hidden off-page text, e.g. bookmark anchors
                chars = [c for c in line if isinstance(c, LTChar)]
                if not chars:
                    continue
                size = median(c.size for c in chars)
                # Drop superscript footnote markers: digits much smaller than the rest of the line
                text = ''.join(c.get_text() for c in line
                               if not (isinstance(c, LTChar) and c.size < 0.7 * size and c.get_text().isdigit()))
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    continue
                top, bottom = page.height - line.y1, page.height - line.y0
                zone = top < page.height * MARGIN_ZONE or bottom > page.height * (1 - MARGIN_ZONE)
                lines.append(Line(text, page_no, line.x0, line.x1, top, bottom, size, zone))
    return lines


def normalize_for_repeat(text):
    return re.sub(r'\d+', '#', text.lower()).strip()


PAGE_NUMBER = re.compile(r'^[\W_]*(page\s*)?(\d+|[ivxlcdm]+)([\W_]*(of\s*\d+)?)[\W_]*$', re.I)


def remove_headers_and_footers(lines, num_pages):
    repeats = Counter(normalize_for_repeat(l.text) for l in lines if l.zone)
    threshold = 3  # each chapter may have its own running header, so a few repeats are enough
    return [l for l in lines
            if not (l.zone and (PAGE_NUMBER.match(l.text) or repeats[normalize_for_repeat(l.text)] >= threshold))]


def median_font_size(lines):
    sizes = [round(l.size, 1) for l in lines for _ in range(max(1, len(l.text) // 20))]
    return median(sizes) if sizes else 10.0


def left_margins(page_lines):
    """The x positions where body lines start (one per column), ignoring indented first lines."""
    counts = Counter(round(l.x0) for l in page_lines)
    margins = []
    for x, n in counts.most_common():
        if n < max(2, len(page_lines) * 0.15):
            break
        if not any(m < x <= m + 40 for m in margins):
            margins.append(x)
    return sorted(margins) or [min(round(l.x0) for l in page_lines)]


TERMINAL = ('.', '!', '?', ':', '”', '"', '’', "'", ')', ']', '—', '…')


def build_paragraphs(lines, body_size):
    paragraphs = []
    current, current_page, current_heading = [], 0, False

    def flush():
        nonlocal current
        if current:
            paragraphs.append(Paragraph(join_lines(current), current_page, current_heading))
        current = []

    pages = {}
    for l in lines:
        pages.setdefault(l.page, []).append(l)

    prev, prev_heading = None, False
    for page_no in sorted(pages):
        page_lines = pages[page_no]
        margins = left_margins(page_lines)
        line_gaps = [b.top - a.bottom for a, b in zip(page_lines, page_lines[1:]) if 0 <= b.top - a.bottom < 3 * a.height]
        typical_gap = median(line_gaps) if line_gaps else body_size * 0.4
        right_edge = max(l.x1 for l in page_lines)
        for l in page_lines:
            is_heading = l.size >= body_size * 1.2 and len(l.text) < 120
            if l.size < body_size * 0.8 and not is_heading:
                continue  # footnotes, captions
            margin = max([m for m in margins if m <= l.x0 + 2] or [margins[0]])
            indented = 3 < l.x0 - margin < 60
            if prev is None:
                new_para = True
            elif is_heading or prev_heading:
                new_para = not (is_heading and prev_heading and l.page == prev.page and l.top - prev.bottom < l.height)
            elif l.page != prev.page:
                # Page break: a paragraph continues unless this line is indented, or the last one clearly ended
                new_para = indented or (prev.text.endswith(TERMINAL) and prev.x1 < right_edge * 0.85)
            else:
                gap = l.top - prev.bottom
                new_para = (indented or gap > typical_gap + 0.6 * l.height or gap < -l.height  # new column
                            or (prev.text.endswith(TERMINAL) and prev.x1 < right_edge * 0.75))
            if new_para:
                flush()
                current_page, current_heading = l.page, is_heading
            current.append(l.text)
            prev, prev_heading = l, is_heading
    flush()
    for p in paragraphs:
        if p.heading and not p.text.endswith(TERMINAL):
            p.text += '.'  # makes the TTS pause after a heading
    return [p for p in paragraphs if p.text.strip()]


def join_lines(texts):
    out = texts[0]
    for t in texts[1:]:
        if re.search(r'\w-$', out) and t[:1].islower():
            out = out[:-1] + t  # undo hyphenation: "jus-" + "tify"
        elif out.endswith(('—', '–', '/')):
            out += t
        else:
            out += ' ' + t
    return out


def split_into_chapters(paragraphs, outline, max_section_chars):
    if outline:
        chapters = chapters_from_outline(paragraphs, outline)
    else:
        chapters = chapters_from_headings(paragraphs)
    chapters = [part for c in chapters for part in cap_length(c, max_section_chars)]
    for c in chapters:
        c.title = c.title or 'Opening'
    return chapters


def chapters_from_outline(paragraphs, outline):
    starts = [(page, title) for _, title, page in outline]
    chapters = [PdfChapter('Front matter')] if paragraphs and paragraphs[0].page < starts[0][0] else []
    idx = 0
    for p in paragraphs:
        while idx < len(starts) and p.page >= starts[idx][0]:
            chapters.append(PdfChapter(starts[idx][1]))
            idx += 1
        chapters[-1].paragraphs.append(p)
    for c in chapters:  # announce the chapter title unless the text already starts with it
        first = c.paragraphs[0].text.lower() if c.paragraphs else ''
        bare_title = re.sub(r'^(chapter|sermon|part|book)\s+[\divxlc]+[.:]?\s*', '', c.title, flags=re.I).lower()
        if c.title != 'Front matter' and bare_title[:40] not in first and c.title.lower()[:40] not in first:
            c.paragraphs.insert(0, Paragraph(c.title.rstrip('.') + '.', c.paragraphs[0].page if c.paragraphs else 0, True))
    return [c for c in chapters if c.paragraphs]


def chapters_from_headings(paragraphs):
    chapters = [PdfChapter('')]
    for p in paragraphs:
        if p.heading and len(chapters[-1].extracted_text) > 500:
            chapters.append(PdfChapter(p.text.rstrip('.')))
        elif p.heading and not chapters[-1].title and not chapters[-1].paragraphs:
            chapters[-1].title = p.text.rstrip('.')
        chapters[-1].paragraphs.append(p)
    return [c for c in chapters if c.paragraphs]


def cap_length(chapter, max_chars):
    if len(chapter.extracted_text) <= max_chars:
        return [chapter]
    parts, current, size = [], [], 0
    for p in chapter.paragraphs:
        if current and size + len(p.text) > max_chars:
            parts.append(current)
            current, size = [], 0
        current.append(p)
        size += len(p.text) + 1
    if current:
        parts.append(current)
    return [PdfChapter(f'{chapter.title} (part {i})' if chapter.title else f'Part {i}', ps)
            for i, ps in enumerate(parts, start=1)]
