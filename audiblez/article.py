# -*- coding: utf-8 -*-
"""Article mode: reads only an article's title, author, date and body, from a PDF of a web page or a magazine article.

Two layers:
  1. rules, always: URLs are removed, and paragraphs that are plainly boilerplate are dropped (cookie and consent
     notices, subscribe/sign-in prompts, ads, share/follow prompts, image credits, "read more" lists, copyright lines);
  2. with an LLM: the LLM reads the numbered paragraphs and says which are the title, byline, date and body, since
     every site lays these out differently and no rule set covers them all.
"""
import json
import re
from types import SimpleNamespace

URL = re.compile(r'\b(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)+/\S*', re.I)
BOILERPLATE = re.compile(r'''^(?:
    advertisement|advertising|sponsored(?:\s+content)?|promoted|paid\s+content|ad\b|
    skip\s+to\s+(?:main\s+)?content|menu|search|home|share(?:\s+this)?|share\s+on\s+\w+|tweet|email|print|save|
    (?:subscribe|sign\s+in|sign\s+up|log\s*in|register|create\s+(?:a\s+free\s+)?account)\b.{0,60}|
    follow\s+(?:us|\w+(?:\s+\w+)?\s+on)\b.*|
    (?:top\s+)?(?:image|photo|picture|video|illustration)\s*(?:credit|courtesy|source)?\s*:.{0,80}|
    (?:getty\s+images|reuters|ap\s+photo|shutterstock)\b.{0,60}|
    (?:read|see|watch|listen)\s+(?:more|also|next)\b.{0,80}|more\s+(?:from|on|stories|like\s+this)\b.{0,60}|
    (?:related|recommended|most\s+(?:read|popular|viewed)|trending|you\s+may\s+also\s+like|editor'?s\s+picks)\b.{0,60}|
    (?:©|copyright\b|all\s+rights\s+reserved).*|terms\s+of\s+(?:use|service).{0,60}|privacy\s+policy.{0,60}|
    \d+\s+comments?|comments?|leave\s+a\s+comment.*|
    (?:\d+\s+)?(?:min(?:ute)?s?|hrs?|hours?|days?)\s+(?:ago|read)\b.{0,10}
)\.?$''', re.I | re.X)
CONSENT = re.compile(r'\b(cookies?|consent|accept\s+(?:all|and\s+continue)|privacy\s+(?:settings|policy|notice|choices)|'
                     r'personali[sz]ed\s+(?:ads|advertising|content)|similar\s+technologies|advertising\s+(?:vendors|partners|faqs?)|'
                     r'(?:collection|use)\s+(?:and\s+use\s+)?of\s+your\s+(?:personal\s+)?(?:information|data)|'
                     r'(?:sharing|share)\s+your\s+(?:personal\s+)?(?:information|data)|choices\s+you\s+(?:may\s+)?have|'
                     r'newsletter|unsubscribe\s+at\s+any\s+time|terms\s+of\s+(?:use|service))\b', re.I)
WEAK_CONSENT_TERMS = {'cookie', 'cookies', 'newsletter'}  # alone, these are as likely to be in an article (a recipe)
IDENTIFIERS = re.compile(r'\b(?:ISBN(?:-1[03])?:?\s*[\d-]{9,17}[\dX]|doi:\s*\S+|ISSN\s*[\d-]{8,9}[\dX]?)\b', re.I)


def strip_urls(text):
    """Removes URLs and catalogue numbers (ISBN, DOI), which nobody wants read aloud."""
    return re.sub(r'\s{2,}', ' ', IDENTIFIERS.sub('', URL.sub('', text))).strip(' ·|-–')


def rule_based(paragraphs):
    """Drops URLs and plainly boilerplate paragraphs. Returns the paragraphs worth considering."""
    kept = []
    for p in paragraphs:
        p = strip_urls(p)
        if not re.search(r'[A-Za-z]{2}', p):
            continue
        consent_terms = {m.lower() for m in CONSENT.findall(p)}
        strong = consent_terms - WEAK_CONSENT_TERMS
        if BOILERPLATE.match(p.strip()) or (len(p) < 400 and len(consent_terms) >= 2) or (len(p) < 160 and strong):
            continue  # a recipe that mentions cookies once is fine; "accept cookies ... consent" is a notice
        kept.append(p)
    return kept


ARTICLE_PROMPT = '''You are preparing a web page or document for an audiobook of its article. You get its numbered
paragraphs (long ones are shortened). Identify the article and nothing else.

Reply with JSON only:
{"title": "the article's headline", "authors": ["author names, as written in the byline"], "date": "publication date
as written, or empty", "body": [[first, last], ...]}

"body" lists the paragraph ranges (inclusive) that make up the article itself: its standfirst/summary, its text,
its subheadings and any quotations or lists inside it. Leave out everything else: navigation, menus, ads, cookie and
privacy notices, subscribe/sign-in prompts, share buttons, image captions and credits, "read more"/related-article
lists, comment sections, author bios at the end, footers, reference lists, bibliographies, "see also" and external
link sections, and tables of contents. If a body paragraph starts with the byline (e.g. "James Landale In the next
week..."), still include it.

Losing part of the article is much worse than reading a stray line, so: include the article's closing paragraphs
(the last paragraph before the footer, related links or author bio usually still belongs to the article), include
quotations, letters and block quotes the article cites, and when you are unsure whether a paragraph belongs to the
article, include it.'''


def llm_select(llm, paragraphs, pdf_title='', log=print):
    numbered = '\n'.join(f'[{i}] {p[:220]}{"…" if len(p) > 220 else ""}' for i, p in enumerate(paragraphs))
    answer = llm.chat_json(ARTICLE_PROMPT, f'Document title from the file: {pdf_title or "unknown"}\n\n{numbered}')
    keep = set()
    for r in answer.get('body') or []:
        if isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(x, int) for x in r):
            keep.update(range(max(0, r[0]), min(len(paragraphs) - 1, r[1]) + 1))
        elif isinstance(r, int) and 0 <= r < len(paragraphs):
            keep.add(r)
    authors = [a for a in (answer.get('authors') or []) if isinstance(a, str) and a.strip()]
    return SimpleNamespace(title=str(answer.get('title') or '').strip(), authors=authors,
                           date=str(answer.get('date') or '').strip(), body=sorted(keep))


def remove_byline(paragraph, authors):
    """'James Landale In the next week...' -> 'In the next week...' (the byline is read in the introduction)."""
    for name in authors:
        m = re.match(rf'^(?:by\s+)?{re.escape(name)}[\s,:–-]+(?=[A-Z“"])', paragraph, flags=re.I)
        if m:
            return paragraph[m.end():]
    return paragraph


def clean_article(book, llm=None, log=print):
    """Turns a loaded book into a single chapter holding only the article: an introduction with its title, author
    and date, then its body. Uses the LLM when given; otherwise only the rules."""
    paragraphs = [p for c in book.chapters for p in c.extracted_text.split('\n') if p.strip()]
    candidates = rule_based(paragraphs)
    log(f'Article mode: {len(paragraphs)} paragraphs, {len(paragraphs) - len(candidates)} dropped as boilerplate by rules')
    title, authors, date = book.title, [book.creator] if book.creator else [], ''
    body = candidates
    if llm is not None and candidates:
        try:
            picked = llm_select(llm, candidates, book.title, log)
            if picked.body:
                body = [candidates[i] for i in fill_gaps(candidates, extend_edges(llm, candidates, picked.body))]
                title, authors, date = picked.title or title, picked.authors or authors, picked.date
                log(f'Article mode: the LLM kept {len(body)} of {len(candidates)} paragraphs as the article body')
            else:
                log('Article mode: the LLM found no article body; keeping everything the rules kept')
        except Exception as e:
            log(f'Article mode: LLM selection failed ({e}); keeping everything the rules kept')
    for i in range(min(3, len(body))):  # the byline is often glued to the first paragraph of text
        cleaned = remove_byline(body[i], authors)
        if cleaned != body[i]:
            body[i] = cleaned
            break
    return article_book(title, authors, date, body, book)


def fill_gaps(candidates, keep):
    """An article has no holes: substantial prose between two kept paragraphs is kept too (a quotation the LLM
    skipped, say). Short lines in between, like captions, stay out."""
    if not keep:
        return keep
    first, last = min(keep), max(keep)
    return sorted(set(keep) | {i for i in range(first, last)
                               if i not in keep and is_prose(candidates[i]) and not CONSENT.search(candidates[i])})


EDGE_PROMPT = '''You check the edge of an article extracted from a web page. You get the article's paragraphs nearest
the edge, and one more paragraph from the page. Does that paragraph belong to the article's own text (its
introduction, its argument, its closing words), or is it website material (a promo, related link, author bio, sign-up
prompt, caption, footer)? Reply with JSON only: {"belongs": true} or {"belongs": false}'''


def extend_edges(llm, candidates, keep, steps=3):
    """Asks about the few paragraphs just after the body's end and before its start, one at a time: the closing
    paragraph of an article is the one an LLM most often leaves out when picking ranges."""
    keep = set(keep)
    for direction in (1, -1):
        edge = max(keep) if direction == 1 else min(keep)
        for _ in range(steps):
            i = edge + direction
            if not (0 <= i < len(candidates)) or len(candidates[i]) < 60 or CONSENT.search(candidates[i]):
                break
            near = sorted(keep)[-2:] if direction == 1 else sorted(keep)[:2]
            where = 'end' if direction == 1 else 'beginning'
            user = (f'The article\'s {where}:\n' + '\n\n'.join(candidates[k][:600] for k in near)
                    + f'\n\nThe paragraph {"after" if direction == 1 else "before"} it on the page:\n{candidates[i][:800]}')
            try:
                if not llm.chat_json(EDGE_PROMPT, user).get('belongs'):
                    break
            except Exception:
                break
            keep.add(i)
            edge = i
    return sorted(keep)


def is_prose(paragraph):
    return len(paragraph) >= 150 and len(re.findall(r'[.!?]["”’)]?(?:\s|$)', paragraph)) >= 1 \
        and len(re.findall(r'\b[a-z]+\b', paragraph)) >= 20


def article_book(title, authors, date, body, book):
    from audiblez.pdf import PdfChapter, Paragraph
    intro = '. '.join(x for x in [title.rstrip('.'), f'By {" and ".join(authors)}' if authors else '', date] if x)
    chapter = PdfChapter(title or 'Article', [Paragraph(intro + '.', 0, True)] + [Paragraph(p, 0) for p in body])
    return SimpleNamespace(title=title, creator=', '.join(authors), cover_image=book.cover_image,
                           chapters=[chapter], good_chapters=[chapter], intro_included=True)
