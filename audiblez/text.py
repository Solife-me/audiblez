# -*- coding: utf-8 -*-
"""Paragraph-aware chunking of chapter text into pieces the TTS models can read well.

A chunk never crosses a paragraph boundary and never splits a sentence unless the sentence alone
is longer than the engine's limit, in which case it is split at a clause (, ; :) or, failing that, a space.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

# Seconds of silence between clips. Clips are trimmed of the silence TTS models pad them with,
# so these are the actual gaps the listener hears.
PARAGRAPH_PAUSE = 1.0  # after a paragraph of narration
DIALOGUE_PARAGRAPH_PAUSE = 0.45  # after a paragraph with dialogue: a turn in a conversation
SENTENCE_PAUSE = 0.3
CLAUSE_PAUSE = 0.12  # e.g. between “My dear Mr. Bennet,” and said his lady


@dataclass
class Chunk:
    text: str
    voice: str
    paragraph_end: bool = False
    pause_after: float = None  # None: decided from paragraph_end and punctuation
    # Chunks of one voice that are synthesized together as a single utterance (for natural intonation),
    # then cut apart: joint holds all the texts of that utterance, joint_index which one this chunk is.
    # A joint text without a chunk (index 0 when joint_context is set) is only there for context.
    joint: tuple = None
    joint_index: int = 0
    joint_context: bool = False

    def pause(self):
        if self.pause_after is not None:
            return self.pause_after
        if self.paragraph_end:
            return PARAGRAPH_PAUSE
        return pause_after_text(self.text)


def pause_after_text(text):
    end = text.rstrip(' ”"’\')]')[-1:] if text.strip() else ''
    if end in '.!?…':
        return SENTENCE_PAUSE
    return CLAUSE_PAUSE


@lru_cache(maxsize=1)
def sentencizer():
    import spacy
    nlp = spacy.blank('xx')  # multi-language tokenizer; the rule-based sentencizer needs no trained model
    nlp.add_pipe('sentencizer')
    return nlp


def split_paragraphs(text):
    return [p.strip() for p in re.split(r'\n+', text) if p.strip()]


# Abbreviations TTS models misread ("Mrs." comes out as anything from "Mrs. Smith" to "M-R-S"), and whose period
# would otherwise be taken for the end of a sentence. Only expanded where they can only mean one thing.
ABBREVIATIONS = [
    (r'\bMrs\.(?=\s)', 'Missus'), (r'\bMr\.(?=\s)', 'Mister'), (r'\bMessrs\.(?=\s)', 'Messieurs'),
    (r'\bMs\.(?=\s+[A-Z])', 'Miz'), (r'\bDr\.(?=\s+[A-Z])', 'Doctor'), (r'\bSt\.(?=\s+[A-Z])', 'Saint'),
    (r'\bRev\.(?=\s+[A-Z])', 'Reverend'), (r'\bCapt\.(?=\s+[A-Z])', 'Captain'), (r'\bCol\.(?=\s+[A-Z])', 'Colonel'),
    (r'\bProf\.(?=\s+[A-Z])', 'Professor'), (r'\b([Cc])hap\.(?=\s+\d)', r'\1hapter'), (r'\b([Vv])ol\.(?=\s+\d)', r'\1olume'), (r'\bMt\.(?=\s+[A-Z])', 'Mount'), (r'&c\.', 'et cetera.'), (r'&c\b', 'et cetera'),
]
ARCHAIC = re.compile(r'\b([Tt])o-(morrow|day|night)\b')  # to-morrow, to-day, to-night (old spellings)
PAGE_MARKER = re.compile(r'\s*\{(?:[ivxlcdm]+|\d+)\}')  # page numbers of the printed edition, e.g. {xii} in Gutenberg texts

# Roman numerals are read letter by letter ("III" as "I I I") by TTS models, but "I" is usually the pronoun and words
# like MIX, DID or CD are spelled with numeral letters, so numerals are only converted where context makes them certain.
ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
CANONICAL_ROMAN = re.compile(r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', re.I)
NUMERAL = r'[IVXLCDM]+|[ivxlcdm]+'
HEADING_WORDS = (r'chapter|chap\.|book|part|volume|vol\.|act|scene|canto|letter|section|sect\.|sermon|psalm|ps\.|'
                 r'appendix|lecture|lesson|article|art\.|stave|epistle|homily|discourse|no\.|number|page|p\.|pp\.')
AFTER_HEADING = re.compile(rf'\b({HEADING_WORDS})(\s+)({NUMERAL})\b(?![\'’]\w)', re.I)
VERSE = re.compile(r'(\d+|' + NUMERAL + r')([.,]?)\s+v\.\s*(\d+)')  # chap. lxiii, v. 10 -> chapter 63, verse 10
CITATION = re.compile(rf'\b([A-Z][a-z]{{1,12}}\.?)\s+({NUMERAL})[.:]\s*(\d+)')  # 1 Cor. xii.12 -> 1 Cor. 12:12
SECTION = re.compile(rf'(^|(?<=[.!?”"’]\s))([IVXL]+)([.,])(?=\s*[A-Z“"‘])')  # "II. Now secondly" at a paragraph or sentence start
LIST = re.compile(rf'\b({NUMERAL})\.(?=,\s*(?:{NUMERAL})\.)|(?<=\.,\s)({NUMERAL})\.')  # contents: "XII., XIII., XIV."
REGNAL_NAMES = ('Alexander|Alfonso|Anne|Benedict|Boniface|Catherine|Charles|Christian|Clement|Constantine|David|'
                'Edward|Elizabeth|Ferdinand|Francis|Frederick|George|Gregory|Gustav|Gustavus|Henry|Innocent|James|'
                'John|Joseph|Leo|Leopold|Louis|Ludwig|Mary|Napoleon|Nicholas|Paul|Peter|Philip|Pius|Ptolemy|'
                'Ramesses|Richard|Robert|Rudolf|Sixtus|Stephen|Urban|Victor|Wilhelm|William')
# a lone I after a name is usually the pronoun ("Elizabeth I went home" is ambiguous), so it needs a clear sign
REGNAL = re.compile(rf'\b({REGNAL_NAMES})\s+((?!I\b)[IVX]+|I(?=\s*(?:[.,;:)]|of\b|and\b|$)))\b(?![\'’]\w)')
NAMED = re.compile(r'\b([A-Z][a-z]+)\s+((?:X{0,3})(?:IX|IV|V?I{1,3})|X{1,3}|V)\b(?![\'’]\w)')  # World War II, Apollo XI
NOT_NUMERALS = {'I', 'MI', 'CD', 'DC', 'CM', 'MD', 'DM', 'MIX', 'DIM', 'DID', 'MID', 'LID', 'MIL', 'ILL', 'CIVIL',
                'VIVID', 'LIVID', 'MIMIC', 'DISC', 'CLI', 'DIV', 'XXX'}
# Abbreviated Bible books in citations ("1 Cor. 12:12", "Isa. 53:5"), only expanded when a chapter number follows,
# so "Rev. Smith" stays Reverend and "Col. Brandon" Colonel
BIBLE_BOOKS = {
    'Gen': 'Genesis', 'Ex': 'Exodus', 'Exod': 'Exodus', 'Lev': 'Leviticus', 'Num': 'Numbers', 'Deut': 'Deuteronomy',
    'Josh': 'Joshua', 'Judg': 'Judges', 'Sam': 'Samuel', 'Kgs': 'Kings', 'Chron': 'Chronicles', 'Neh': 'Nehemiah',
    'Esth': 'Esther', 'Ps': 'Psalm', 'Psa': 'Psalm', 'Prov': 'Proverbs', 'Eccl': 'Ecclesiastes', 'Eccles': 'Ecclesiastes',
    'Isa': 'Isaiah', 'Jer': 'Jeremiah', 'Lam': 'Lamentations', 'Ezek': 'Ezekiel', 'Dan': 'Daniel', 'Hos': 'Hosea',
    'Obad': 'Obadiah', 'Mic': 'Micah', 'Nah': 'Nahum', 'Hab': 'Habakkuk', 'Zeph': 'Zephaniah', 'Hag': 'Haggai',
    'Zech': 'Zechariah', 'Mal': 'Malachi', 'Matt': 'Matthew', 'Mk': 'Mark', 'Lk': 'Luke', 'Jn': 'John', 'Rom': 'Romans',
    'Cor': 'Corinthians', 'Gal': 'Galatians', 'Eph': 'Ephesians', 'Phil': 'Philippians', 'Col': 'Colossians',
    'Thess': 'Thessalonians', 'Tim': 'Timothy', 'Tit': 'Titus', 'Philem': 'Philemon', 'Heb': 'Hebrews', 'Jas': 'James',
    'Pet': 'Peter', 'Rev': 'Revelation'}
NUMBERED_BOOKS = 'Samuel|Kings|Chronicles|Corinthians|Thessalonians|Timothy|Peter|John'
BIBLE_ABBREVIATION = re.compile(r'(?:\b([123])\s+)?\b(' + '|'.join(sorted(BIBLE_BOOKS, key=len, reverse=True)) + r')\.\s*(?=\d)')
BIBLE_NUMBERED = re.compile(rf'\b([123])\s+({NUMBERED_BOOKS})\b(?=\s+\d)')
BIBLE_VERSE = re.compile(r'\b(' + '|'.join(sorted(set(BIBLE_BOOKS.values()))) + r')\s+(\d+)\.\s*(\d+)\b')  # Psalm 23. 1 -> Psalm 23:1
ORDINALS = ['', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth', 'Eleventh',
            'Twelfth', 'Thirteenth', 'Fourteenth', 'Fifteenth', 'Sixteenth', 'Seventeenth', 'Eighteenth', 'Nineteenth',
            'Twentieth']


def roman_to_int(numeral):
    """The value of a canonical Roman numeral (any case), or None."""
    if not numeral or not CANONICAL_ROMAN.match(numeral):
        return None
    numeral = numeral.upper()
    total = 0
    for i, ch in enumerate(numeral):
        value = ROMAN_VALUES[ch]
        total += -value if i + 1 < len(numeral) and ROMAN_VALUES[numeral[i + 1]] > value else value
    return total


def roman_numerals_to_numbers(text):
    def number(numeral, fallback):
        value = roman_to_int(numeral)
        return str(value) if value else fallback

    def after_heading(m):
        word, space, numeral = m.group(1), m.group(2), m.group(3)
        if len(numeral) == 1:  # "the book I love", "for the most part I can": a lone letter needs a clear heading
            heading = word[:1].isupper() or word.endswith('.')
            followed = re.match(r'\s*(?:[.,;:)\]]|$)', text[m.end():])
            if not (heading and followed):
                return m.group(0)
        return word + space + number(numeral, numeral)

    text = AFTER_HEADING.sub(after_heading, text)
    text = VERSE.sub(lambda m: f'{number(m.group(1), m.group(1))}, verse {m.group(3)}', text)
    text = CITATION.sub(lambda m: f'{m.group(1)} {m.group(2)}:{m.group(3)}' if roman_to_int(m.group(2)) is None
                        else f'{m.group(1)} {roman_to_int(m.group(2))}:{m.group(3)}', text)
    text = SECTION.sub(lambda m: m.group(1) + number(m.group(2), m.group(2)) + m.group(3)
                       if m.group(2) not in NOT_NUMERALS - {'I'} else m.group(0), text)
    text = LIST.sub(lambda m: number(m.group(1) or m.group(2), m.group(1) or m.group(2)) + '.', text)
    text = REGNAL.sub(lambda m: f'{m.group(1)} the {ORDINALS[roman_to_int(m.group(2))]}'
                      if (roman_to_int(m.group(2)) or 99) <= 20 else m.group(0), text)
    text = NAMED.sub(lambda m: m.group(0) if m.group(2) in NOT_NUMERALS or m.group(1) in ('I', 'A')
                     else f'{m.group(1)} {roman_to_int(m.group(2))}', text)
    return text


def normalize_for_speech(text, roman_numerals=True):
    text = PAGE_MARKER.sub('', text)
    if roman_numerals:
        text = roman_numerals_to_numbers(text)  # before abbreviations: "chap.", "Ps." and "St." are context here
    text = BIBLE_ABBREVIATION.sub(lambda m: (ORDINALS[int(m.group(1))] + ' ' if m.group(1) else '') + BIBLE_BOOKS[m.group(2)] + ' ', text)
    text = BIBLE_NUMBERED.sub(lambda m: f'{ORDINALS[int(m.group(1))]} {m.group(2)}', text)
    text = BIBLE_VERSE.sub(r'\1 \2:\3', text)
    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    return ARCHAIC.sub(lambda m: m.group(1) + 'o' + m.group(2), text)


def split_sentences(paragraph):
    nlp = sentencizer()
    if len(paragraph) > nlp.max_length:
        nlp.max_length = len(paragraph) + 1
    return [s.text.strip() for s in nlp(paragraph).sents if s.text.strip()]


def split_long_sentence(text, max_length=400):
    """Split a sentence longer than max_length, preferring clause boundaries, then whitespace."""
    if len(text) <= max_length:
        return [text]
    parts = []
    while len(text) > max_length:
        window = text[:max_length]
        split_index = max(window.rfind(p) for p in (', ', '; ', ': ', ' – ', ' — '))
        if split_index < max_length // 3:  # a clause break too early would leave a tiny fragment
            split_index = window.rfind(' ')
        else:
            split_index += 1  # keep the punctuation with the first part
        if split_index <= 0:
            split_index = max_length
        parts.append(text[:split_index].strip())
        text = text[split_index:].strip()
    if text:
        parts.append(text)
    return parts


def pack_sentences(sentences, max_chars):
    """Greedily packs whole sentences into chunks of at most max_chars."""
    chunks, current = [], ''
    for sent in sentences:
        for piece in split_long_sentence(sent, max_chars):
            if current and len(current) + 1 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f'{current} {piece}' if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_paragraph(paragraph, voice, max_chars, roman_numerals=True):
    paragraph = normalize_for_speech(paragraph, roman_numerals)
    if len(paragraph) <= max_chars:
        pieces = [paragraph]
    else:
        pieces = pack_sentences(split_sentences(paragraph), max_chars)
    chunks = [Chunk(p, voice) for p in pieces]
    if chunks:
        chunks[-1].paragraph_end = True
    return chunks


def chunk_text(text, voice, max_chars, roman_numerals=True):
    """Chunks for narrating text in a single voice."""
    return [c for p in split_paragraphs(text) for c in chunk_paragraph(p, voice, max_chars, roman_numerals)]
