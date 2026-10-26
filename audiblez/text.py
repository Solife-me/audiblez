# -*- coding: utf-8 -*-
"""Paragraph-aware chunking of chapter text into pieces the TTS models can read well.

A chunk never crosses a paragraph boundary and never splits a sentence unless the sentence alone
is longer than the engine's limit, in which case it is split at a clause (, ; :) or, failing that, a space.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

PARAGRAPH_PAUSE = 0.45  # seconds of silence after each paragraph
SENTENCE_PAUSE = 0.0


@dataclass
class Chunk:
    text: str
    voice: str
    paragraph_end: bool = False


@lru_cache(maxsize=1)
def sentencizer():
    import spacy
    nlp = spacy.blank('xx')  # multi-language tokenizer; the rule-based sentencizer needs no trained model
    nlp.add_pipe('sentencizer')
    return nlp


def split_paragraphs(text):
    return [p.strip() for p in re.split(r'\n+', text) if p.strip()]


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
