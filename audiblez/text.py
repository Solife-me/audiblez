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
    (r'\bProf\.(?=\s+[A-Z])', 'Professor'), (r'\bMt\.(?=\s+[A-Z])', 'Mount'), (r'&c\.', 'et cetera.'), (r'&c\b', 'et cetera'),
]
ARCHAIC = re.compile(r'\b([Tt])o-(morrow|day|night)\b')  # to-morrow, to-day, to-night (old spellings)
NUMBERED = re.compile(r'\b(Chapter|Book|Part|Volume|Act|Scene|Canto|Letter|Section|Sermon|Psalm)\s+([IVXLC]+)\b(?![\'’]\w)')
ROMAN = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}


def roman_to_int(numeral):
    total = 0
    for i, ch in enumerate(numeral):
        value = ROMAN[ch]
        total += -value if i + 1 < len(numeral) and ROMAN[numeral[i + 1]] > value else value
    return total


def normalize_for_speech(text):
    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    text = ARCHAIC.sub(lambda m: m.group(1) + 'o' + m.group(2), text)
    return NUMBERED.sub(lambda m: f'{m.group(1)} {roman_to_int(m.group(2))}', text)


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


def chunk_paragraph(paragraph, voice, max_chars):
    paragraph = normalize_for_speech(paragraph)
    if len(paragraph) <= max_chars:
        pieces = [paragraph]
    else:
        pieces = pack_sentences(split_sentences(paragraph), max_chars)
    chunks = [Chunk(p, voice) for p in pieces]
    if chunks:
        chunks[-1].paragraph_end = True
    return chunks


def chunk_text(text, voice, max_chars):
    """Chunks for narrating text in a single voice."""
    return [c for p in split_paragraphs(text) for c in chunk_paragraph(p, voice, max_chars)]
