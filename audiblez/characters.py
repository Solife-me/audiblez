# -*- coding: utf-8 -*-
"""Multi-voice narration: detects who speaks each quotation and casts a voice for every character.

Flow:
  1. split every paragraph into narration and quotation spans (curly, straight or British single quotes),
  2. ask an LLM, a window of paragraphs at a time, who speaks each quotation, keeping a running character list,
  3. merge name variants ("Darcy" / "Mr. Darcy"), count lines, and assign voices by gender, accent and age,
  4. write an editable cast sheet (<book>.cast.toml); attributions are cached in <book>.speakers.json.
"""
import hashlib
import json
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from audiblez.text import Chunk, split_paragraphs, pack_sentences, split_sentences, normalize_for_speech, \
    PARAGRAPH_PAUSE, DIALOGUE_PARAGRAPH_PAUSE

NARRATOR = 'narrator'
UNKNOWN = 'unknown'
WINDOW_CHARS = 3000
CONTEXT_PARAGRAPHS = 3
MALE_TITLES = {'mr', 'mister', 'messrs', 'sir', 'lord', 'master', 'king', 'prince', 'duke', 'father', 'brother', 'uncle', 'monsieur', 'herr', 'señor'}
FEMALE_TITLES = {'mrs', 'missus', 'miss', 'ms', 'miz', 'lady', 'madam', 'madame', 'dame', 'queen', 'princess', 'duchess', 'mother', 'sister', 'aunt', 'mademoiselle', 'frau', 'señora'}
NEUTRAL_TITLES = {'dr', 'doctor', 'professor', 'prof', 'captain', 'colonel', 'general', 'rev', 'reverend', 'the', 'old', 'young', 'little'}
ACCENTS = ['american', 'british', 'irish', 'scottish', 'welsh', 'australian', 'canadian', 'indian', 'south african',
           'french', 'german', 'italian', 'spanish', 'other']


@dataclass
class Span:
    text: str
    is_quote: bool


# ---------------------------------------------------------------------------------------------------------------
# Quotation detection

def detect_quote_style(text):
    curly = text.count('“')
    straight = text.count('"') // 2
    single = len(re.findall(r'(?:^|[\s(\[—–-])‘(?=\S)', text, flags=re.M))
    style = max([('double', curly), ('straight', straight), ('single', single)], key=lambda s: s[1])
    return style[0] if style[1] else 'double'


def split_dialogue(paragraph, style):
    """Splits a paragraph into narration and quotation spans. A quote left open runs to the end of the paragraph,
    as with dialogue that continues into the next paragraph."""
    spans, buf, in_quote = [], '', False

    def push(is_quote):
        nonlocal buf
        if buf:
            spans.append(Span(buf, is_quote))
        buf = ''

    n = len(paragraph)
    for i, ch in enumerate(paragraph):
        prev = paragraph[i - 1] if i else ' '
        nxt = paragraph[i + 1] if i + 1 < n else ' '
        if style == 'double':
            opening, closing = ch == '“', ch == '”'
        elif style == 'straight':
            opening, closing = ch == '"' and not in_quote, ch == '"' and in_quote
        else:  # British single quotes; ’ between letters is an apostrophe (don’t), after s it may be possessive
            opening = ch == '‘' and (prev.isspace() or prev in '([—–-“') and not nxt.isspace()
            after = paragraph[i + 2] if i + 2 < n else ''
            possessive = prev == 's' and nxt == ' ' and after.islower()  # the boys’ turn
            closing = ch == '’' and in_quote and not nxt.isalnum() and not prev.isspace() and not possessive
        if opening and not in_quote:
            push(False)
            buf, in_quote = ch, True
        elif closing and in_quote:
            buf += ch
            push(True)
            in_quote = False
        else:
            buf += ch
    push(in_quote)
    return spans


def paragraph_spans(text, style):
    return [split_dialogue(p, style) for p in split_paragraphs(text)]


# ---------------------------------------------------------------------------------------------------------------
# Characters

AGES = ('child', 'young', 'adult', 'middle-aged', 'elderly')
AGE_YEARS = {'child': 10, 'young': 20, 'adult': 35, 'middle-aged': 52, 'elderly': 70}


@dataclass
class Character:
    name: str
    gender: str = 'unknown'
    age: str = 'adult'
    aliases: set = field(default_factory=set)
    lines: int = 0
    voice: str = ''
    description: str = ''  # personality and way of speaking, from the LLM
    why: str = ''  # why this voice was chosen


def name_tokens(name):
    return [t for t in re.sub(r'[^\w\s-]', ' ', name.lower()).split()]


def title_gender(name):
    tokens = set(name_tokens(name))
    if tokens & MALE_TITLES: return 'male'
    if tokens & FEMALE_TITLES: return 'female'
    return None


class Registry:
    def __init__(self, characters=None):
        self.characters = {c.name: c for c in (characters or [])}

    def known(self):
        return [{'name': c.name, 'gender': c.gender, **({'aliases': sorted(c.aliases)} if c.aliases else {})}
                for c in sorted(self.characters.values(), key=lambda c: -c.lines)]

    def resolve(self, name):
        """Maps a speaker name from the LLM to a canonical character name (or None if new)."""
        low = name.strip().lower()
        if low in (NARRATOR, UNKNOWN, ''):
            return low or UNKNOWN
        for c in self.characters.values():
            if low == c.name.lower() or low in {a.lower() for a in c.aliases}:
                return c.name
        tokens = set(name_tokens(name)) - MALE_TITLES - FEMALE_TITLES - NEUTRAL_TITLES
        if not tokens:
            return None
        gender = title_gender(name)
        candidates = []
        for c in self.characters.values():
            c_tokens = set(name_tokens(c.name)) - MALE_TITLES - FEMALE_TITLES - NEUTRAL_TITLES
            c_gender = title_gender(c.name) or (c.gender if c.gender != 'unknown' else None)
            if gender and c_gender and gender != c_gender:
                continue  # Mr. Bennet is not Mrs. Bennet
            if tokens <= c_tokens or (c_tokens and c_tokens <= tokens):
                candidates.append(c)
        if len(candidates) == 1:
            if len(name_tokens(name)) > len(name_tokens(candidates[0].name)):
                self.rename(candidates[0], name.strip())  # prefer the fuller name
            candidates[0].aliases.add(name.strip())
            return candidates[0].name
        return None

    def rename(self, character, new_name):
        del self.characters[character.name]
        character.aliases.add(character.name)
        character.aliases.discard(new_name)
        character.name = new_name
        self.characters[new_name] = character

    def add(self, name, gender='unknown', age='adult', aliases=(), description=''):
        name = name.strip()
        existing = self.resolve(name)
        if existing and existing not in (NARRATOR, UNKNOWN):
            c = self.characters[existing]
        else:
            c = self.characters.setdefault(name, Character(name))
        if c.gender == 'unknown' and gender in ('male', 'female'):
            c.gender = gender
        if c.gender == 'unknown':
            c.gender = title_gender(name) or 'unknown'
        if age in AGES and age != 'adult':
            c.age = age
        if description and not c.description:
            c.description = str(description).strip()[:120]
        for alias in (a.strip() for a in aliases if a and a.strip()):
            if alias.lower() == c.name.lower():
                continue
            other = self.characters.get(alias)
            if other is not None and other is not c:  # the same person was first seen under this alias: merge
                del self.characters[alias]
                c.aliases |= other.aliases
                if c.gender == 'unknown':
                    c.gender = other.gender
            c.aliases.add(alias)
        return c.name


# ---------------------------------------------------------------------------------------------------------------
# Speaker attribution with the LLM

SYSTEM_PROMPT = '''You label who speaks each quotation in a passage of a book, for a multi-voice audiobook.

Rules:
- Every quotation is marked with an id like [Q12] just before its opening quote mark. Label every id in the passage.
- Answer with the character who says the quotation out loud (or thinks it, if written as speech).
- Use the fullest name the text or the known-character list gives (e.g. "Elizabeth Bennet", not "Lizzy").
  Reuse names from the known-character list exactly, character for character. Keep titles that distinguish
  people (Mr. Bennet and Mrs. Bennet are different characters).
- Use "narrator" when the quotation is not a character speaking: quoted writing, letters read by the narrator,
  signs, titles, scare quotes, proverbs, or the narrator quoting something.
- Use "unknown" if a character speaks but the text gives no way to tell who.
- Speakers usually alternate paragraph by paragraph in a conversation, and quotes in the same paragraph are
  usually the same speaker. Use the previous context, where speakers are already labeled like [Name]“...”.
- List every speaker you name who is not in the known-character list under "characters", with gender
  ("male", "female" or "unknown", from names, pronouns and titles), age ("child", "young", "adult",
  "middle-aged" or "elderly", from what the text implies), a short description of their personality and way of
  speaking (e.g. "excitable, talkative mother" or "dry, sarcastic old gentleman"), and any other names the passage
  uses for them.

Reply with JSON only, in this shape:
{"quotes": {"12": "Name", "13": "narrator"}, "characters": [{"name": "Name", "gender": "female", "age": "adult",
 "description": "excitable, talkative mother", "aliases": []}]}'''


def render_paragraph(spans, quote_ids=None, speakers=None):
    out = []
    qi = 0
    for s in spans:
        if s.is_quote:
            if quote_ids is not None:
                out.append(f'[Q{quote_ids[qi]}]{s.text}')
            elif speakers is not None:
                out.append(f'[{speakers[qi]}]{s.text}')
            else:
                out.append(s.text)
            qi += 1
        else:
            out.append(s.text)
    return ''.join(out)


def attribute_chapter(llm, spans_by_paragraph, registry, log=print):
    """Returns the speaker (canonical name, 'narrator' or 'unknown') of every quote in the chapter, in order."""
    quote_counts = [sum(s.is_quote for s in spans) for spans in spans_by_paragraph]
    speakers = [UNKNOWN] * sum(quote_counts)
    first_quote = [sum(quote_counts[:i]) for i in range(len(quote_counts))]
    windows, current, size = [], [], 0
    for i, spans in enumerate(spans_by_paragraph):
        length = sum(len(s.text) for s in spans)
        if current and size + length > WINDOW_CHARS:
            windows.append(current)
            current, size = [], 0
        current.append(i)
        size += length
    if current:
        windows.append(current)

    for w, paragraph_ids in enumerate(windows):
        ids = [q for i in paragraph_ids for q in range(first_quote[i], first_quote[i] + quote_counts[i])]
        if not ids:
            continue
        context_ids = range(max(0, paragraph_ids[0] - CONTEXT_PARAGRAPHS), paragraph_ids[0])
        context = '\n\n'.join(render_paragraph(spans_by_paragraph[i], speakers=speakers[first_quote[i]:first_quote[i] + quote_counts[i]])
                              for i in context_ids)
        passage = '\n\n'.join(render_paragraph(spans_by_paragraph[i], quote_ids=list(range(first_quote[i], first_quote[i] + quote_counts[i])))
                              for i in paragraph_ids)
        user = (f'Known characters: {json.dumps(registry.known(), ensure_ascii=False)}\n\n'
                + (f'Previous context (already labeled):\n{context}\n\n' if context else '')
                + f'Passage:\n{passage}')
        log(f'  Attributing quotes {ids[0]}-{ids[-1]} (window {w + 1}/{len(windows)})...')
        try:
            answer = llm.chat_json(SYSTEM_PROMPT, user)
        except Exception as e:
            log(f'  Speaker detection failed for this window ({e}); these quotes will be read by the narrator.')
            continue
        for c in answer.get('characters') or []:
            if isinstance(c, dict) and c.get('name') and c['name'].strip().lower() not in (NARRATOR, UNKNOWN):
                registry.add(c['name'], c.get('gender', 'unknown'), c.get('age', 'adult'), c.get('aliases') or [],
                             c.get('description', ''))
        quotes = answer.get('quotes') or {}
        for q in ids:
            name = str(quotes.get(str(q)) or quotes.get(f'Q{q}') or UNKNOWN).strip()
            resolved = registry.resolve(name)
            if resolved is None:
                resolved = registry.add(name)
            speakers[q] = resolved
    # Characters may have been merged along the way (Lizzy -> Elizabeth): map every speaker to its final name
    return [registry.resolve(s) or s for s in speakers]


# ---------------------------------------------------------------------------------------------------------------
# Narrator suggestion (author / narrator origin -> accent)

NARRATOR_PROMPT = f'''You help choose the narrator voice for an audiobook. Given the title, the author and the opening of a
book, decide:
- "narrator_gender": the gender of the first-person narrator if the book is written in the first person,
  otherwise the author's gender ("male", "female" or "unknown").
- "accent": the accent that best fits the author's or narrator's origin and the setting, one of {ACCENTS}.
- "confidence": "high" only if you actually know this author or the text makes it clear, otherwise "low".
- "reason": one short sentence.
Reply with JSON only: {{"narrator_gender": "...", "accent": "...", "confidence": "...", "reason": "..."}}'''


def suggest_narrator(llm, title, creator, opening_text):
    user = f'Title: {title or "unknown"}\nAuthor: {creator or "unknown"}\n\nOpening:\n{opening_text[:1500]}'
    answer = llm.chat_json(NARRATOR_PROMPT, user)
    accent = str(answer.get('accent', 'other')).lower()
    accent = {'english': 'british', 'uk': 'british', 'us': 'american', 'usa': 'american'}.get(accent, accent)
    return dict(gender=str(answer.get('narrator_gender', 'unknown')).lower(), accent=accent,
                confidence=str(answer.get('confidence', 'low')).lower(), reason=answer.get('reason', ''))


def pick_narrator_voice(voices, suggestion, current_voice):
    """Returns a voice matching the suggested gender and accent, or None. Keeps current_voice if it already fits."""
    def fits(v):
        return (v.gender == suggestion['gender'] and v.accent == suggestion['accent']
                and getattr(v, 'style', '') != 'character')
    if current_voice in voices and fits(voices[current_voice]):
        return current_voice
    matches = [v.name for v in voices.values() if fits(v)]
    return matches[0] if matches else None


def auto_pick_narrator(llm, book, chapters, engine, current_voice, log=print, with_note=False):
    """Picks a narrator voice that matches the author's (or first-person narrator's) gender and origin.
    Only switches voice on a high-confidence suggestion with a matching tagged voice."""
    opening = next((c.extracted_text for c in chapters if len(c.extracted_text) > 500), '')
    suggestion = suggest_narrator(llm, book.title, book.creator, opening)
    note = (f'Narrator suggestion: {suggestion["gender"]}, {suggestion["accent"]} accent '
            f'({suggestion["confidence"]} confidence): {suggestion["reason"]}')
    log(note)
    voice = current_voice
    picked = pick_narrator_voice(engine.list_voices(), suggestion, current_voice) if suggestion['confidence'] == 'high' else None
    if picked:
        voice = picked
        log(f'Narrator voice: {voice}')
    else:
        log(f'Keeping narrator voice {voice} (no confident match among the tagged voices).')
    return (voice, note) if with_note else voice


# ---------------------------------------------------------------------------------------------------------------
# Casting

def eligible_voices(voices, narrator_voice, engine_name):
    pool = [v for v in voices.values() if v.name != narrator_voice and getattr(v, 'style', '') != 'character']
    if engine_name == 'kokoro':  # a Kokoro voice can only read its own language; american/british mix fine
        group = 'ab' if narrator_voice[0] in 'ab' else narrator_voice[0]
        pool = [v for v in pool if v.name[0] in group]
    return pool


def assign_voices(characters, voices, narrator_voice, engine_name, accent=None):
    """Gives each character (most lines first) the least used voice of their gender, preferring the book's accent,
    the closest age and then the most expressive voice, so the characters who talk most get the liveliest voices.
    Characters of unknown gender keep the narrator's voice. Characters that already have a voice keep it."""
    pool = eligible_voices(voices, narrator_voice, engine_name)
    used = Counter(c.voice for c in characters if c.voice)
    for c in sorted(characters, key=lambda c: -c.lines):
        if c.voice:
            continue
        candidates = [v for v in pool if v.gender == c.gender]
        if c.gender not in ('male', 'female') or not candidates:
            c.voice = narrator_voice
            continue
        def score(v):
            if getattr(v, 'age_years', 0):
                age_gap = round(abs(v.age_years - AGE_YEARS.get(c.age, 35)) / 12)
            else:
                age_gap = 0 if getattr(v, 'age', 'adult') == c.age or (c.age == 'child' and v.age == 'young') else 1
            return (used[v.name], 0 if accent and v.accent == accent else 1, age_gap, -getattr(v, 'expressiveness', 0.0))
        best = min(candidates, key=score)
        c.voice = best.name
        used[best.name] += 1


CASTING_PROMPT = """You are casting voices for a multi-voice audiobook. Assign one voice to every character under
"Characters to cast", choosing from "Voices".

Rules, most important first:
1. Gender: a character must get a voice of the same gender.
2. Age: match the character's age to the voice's apparent age (age_years). Children and teenagers get the youngest
   voices; elderly characters get the oldest. Never give a young character an old voice or the other way round.
3. Distinct voices: the characters with the most lines must each get a voice that no other character uses, and
   characters who talk to each other should never share a voice. Only minor characters (a handful of lines) may
   share a voice, and never with a main character. Characters under "Already cast" keep their voices: count them.
4. Temperament: lively, emotional, excitable or comic characters get the most expressive voices (high
   expressiveness); calm, dry, stern or reserved characters get steadier voices (lower expressiveness). Deep voices
   (low pitch) suit authority, gravity or menace; lighter voices suit youth and warmth.
5. Accent: characters usually share the accent of the book's setting, and the narrator's accent is a good guide.
   Give a different accent only to characters the description marks as foreign or from elsewhere.
6. Never use the narrator's voice. Voices with style "character" are novelty voices for creatures and fairy-tale
   figures: use one only for a character of that kind (a dragon, an ogre, a fairy, a monster).

Reply with JSON only: {"cast": {"Character name": {"voice": "Voice name", "why": "a few words"}}}"""


def voice_profile(v):
    profile = {'name': v.name, 'gender': v.gender, 'age': v.age, 'accent': v.accent}
    for key in ('age_years', 'pitch', 'expressiveness'):
        if getattr(v, key, 0):
            profile[key] = getattr(v, key)
    if getattr(v, 'style', ''):
        profile['style'] = v.style
    return profile


def llm_cast(llm, characters, voices, narrator_voice, engine_name, book=None, log=print):
    """Lets the LLM cast the characters without a voice, using the voice profiles and character descriptions.
    Returns how many were cast; anything invalid is left for assign_voices."""
    todo = [c for c in sorted(characters, key=lambda c: -c.lines) if not c.voice and c.gender in ('male', 'female')]
    if not todo:
        return 0
    pool = [v for v in voices.values() if v.name != narrator_voice]
    if engine_name == 'kokoro':
        group = 'ab' if narrator_voice[0] in 'ab' else narrator_voice[0]
        pool = [v for v in pool if v.name[0] in group]
    describe = lambda c: {'name': c.name, 'gender': c.gender, 'age': c.age, 'lines': c.lines,
                          **({'description': c.description} if c.description else {})}
    narrator = voices.get(narrator_voice)
    user = '\n\n'.join([
        f'Book: {getattr(book, "title", "") or "unknown"} by {getattr(book, "creator", "") or "unknown"}',
        f'Narrator voice: {json.dumps(voice_profile(narrator) if narrator else {"name": narrator_voice})}',
        f'Already cast: {json.dumps([{**describe(c), "voice": c.voice} for c in characters if c.voice], ensure_ascii=False)}',
        f'Characters to cast: {json.dumps([describe(c) for c in todo], ensure_ascii=False)}',
        f'Voices: {json.dumps([voice_profile(v) for v in pool], ensure_ascii=False)}'])
    try:
        answer = llm.chat_json(CASTING_PROMPT, user).get('cast') or {}
    except Exception as e:
        log(f'LLM casting failed ({e}); casting by voice profile instead.')
        return 0
    by_name = {v.name: v for v in pool}
    cast = 0
    for c in todo:
        choice = answer.get(c.name) or next((a for n, a in answer.items() if n.lower() == c.name.lower()), None)
        voice = choice.get('voice') if isinstance(choice, dict) else choice
        v = by_name.get(voice)
        if v is None or (v.gender in ('male', 'female') and v.gender != c.gender):
            if voice:
                log(f'  Ignoring the LLM\'s choice of "{voice}" for {c.name} (not a {c.gender} voice).')
            continue
        c.voice = v.name
        c.why = str(choice.get('why', '')).strip()[:100] if isinstance(choice, dict) else ''
        cast += 1
    # Rule 3 is the one models break most: a main character must not share a voice with a character who has more lines
    seen = {}
    for c in sorted(characters, key=lambda c: -c.lines):
        if c.voice and c.voice in seen and c.lines >= 5 and c in todo:
            log(f'  {c.name} would share {c.voice} with {seen[c.voice]}; picking another voice.')
            c.voice, c.why = '', ''
            cast -= 1
        elif c.voice:
            seen.setdefault(c.voice, c.name)
    return cast


# ---------------------------------------------------------------------------------------------------------------
# Cast sheet and attribution cache

def toml_str(s):
    return json.dumps(str(s), ensure_ascii=False)


def write_cast_sheet(path, title, creator, narrator_voice, characters, narrator_note='', voices=None):
    lines = [f'# audiblez cast sheet for {title or "this book"}{f" by {creator}" if creator else ""}',
             '# Change any voice (see `audiblez --list-voices` for each voice\'s gender, age, accent and expressiveness),',
             '# then run the same command again to make the audiobook.',
             '# Set a character\'s voice to the narrator\'s voice to have the narrator read their lines.',
             '# Delete this file and the .speakers.json next to it to detect the characters again.', '']
    if narrator_note:
        lines += [f'# {narrator_note}']
    lines += [f'narrator = {toml_str(narrator_voice)}', '']
    for c in sorted(characters, key=lambda c: -c.lines):
        v = (voices or {}).get(c.voice)
        profile = f'  # {v.gender}, {v.age}, {v.accent}, expressiveness {v.expressiveness}' if v and v.expressiveness else ''
        lines += [f'[characters.{toml_str(c.name)}]', f'voice = {toml_str(c.voice)}{profile}', f'gender = {toml_str(c.gender)}']
        if c.age != 'adult':
            lines.append(f'age = {toml_str(c.age)}')
        if c.description:
            lines.append(f'description = {toml_str(c.description)}')
        if c.why:
            lines.append(f'why = {toml_str(c.why)}')
        lines.append(f'lines = {c.lines}')
        if c.aliases:
            lines.append(f'aliases = [{", ".join(toml_str(a) for a in sorted(c.aliases))}]')
        lines.append('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def read_cast_sheet(path):
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    characters = [Character(name, c.get('gender', 'unknown'), c.get('age', 'adult'), set(c.get('aliases', [])),
                            c.get('lines', 0), c.get('voice', ''), c.get('description', ''), c.get('why', ''))
                  for name, c in data.get('characters', {}).items()]
    return data.get('narrator'), characters


def text_key(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


class Cast:
    """Everything needed to voice a book: narrator, characters and the speaker of every quote."""

    def __init__(self, cast_path, cache_path, narrator_voice, registry, style, speakers_by_chapter):
        self.cast_path, self.cache_path = Path(cast_path), Path(cache_path)
        self.narrator_voice = narrator_voice
        self.registry = registry
        self.style = style
        self.speakers_by_chapter = speakers_by_chapter

    def voice_for(self, speaker):
        if speaker in (NARRATOR, UNKNOWN):
            return self.narrator_voice
        c = self.registry.characters.get(speaker) or self.registry.characters.get(self.registry.resolve(speaker) or '')
        return c.voice if c and c.voice else self.narrator_voice

    def signature(self):
        """Short hash of the casting, so chapter audio is regenerated when the cast sheet changes."""
        items = sorted((c.name, c.voice) for c in self.registry.characters.values())
        return hashlib.sha1(json.dumps([self.narrator_voice, items]).encode()).hexdigest()[:8]

    def chunks(self, text, max_chars, prime_replies=False, **speech):
        """Chunks for one chapter: narration in the narrator's voice, each quote in its speaker's voice.

        Quotes of one speaker split by a short dialogue tag ("“My dear,” said she, “have you heard?”") are
        synthesized as one utterance so the first part keeps its continuing intonation. With prime_replies, a
        reply that opens a paragraph is synthesized after the line it answers (in the replying voice; that part
        is cut off again), so it is spoken as a response rather than out of nowhere."""
        spans_by_paragraph = paragraph_spans(text, self.style)
        speakers = self.speakers_by_chapter.get(text_key(text), [])
        chunks, qi = [], 0
        last_line = None  # (voice, text) of the latest quote, the line a reply answers
        for spans in spans_by_paragraph:
            voiced = []  # [voice, text, is_quote], merging neighbours that share a voice
            for s in spans:
                if s.is_quote:
                    voice = self.voice_for(speakers[qi] if qi < len(speakers) else UNKNOWN)
                    qi += 1
                else:
                    voice = self.narrator_voice
                if voiced and voiced[-1][0] == voice:
                    voiced[-1][1] += s.text
                    voiced[-1][2] = voiced[-1][2] or s.is_quote
                else:
                    voiced.append([voice, s.text, s.is_quote])
            voiced = [(v, normalize_for_speech(t.strip(), **speech), q) for v, t, q in voiced
                      if re.search(r'\w', t)]  # drop stray punctuation
            has_dialogue = any(q for _, _, q in voiced)

            # Group each speaker's quotes that are separated only by a short narrator tag
            groups, member = [], {}
            for idx, (voice, part, is_quote) in enumerate(voiced):
                if not is_quote or voice == self.narrator_voice or idx in member:
                    continue
                group = [idx]
                while (group[-1] + 2 < len(voiced) and len(voiced[group[-1] + 1][1]) < 100
                       and voiced[group[-1] + 2][0] == voice and voiced[group[-1] + 2][2]):
                    group.append(group[-1] + 2)
                context = None
                if prime_replies and idx == 0 and last_line and last_line[0] != voice:
                    context = last_sentence(last_line[1])
                texts = ([context] if context else []) + [voiced[k][1] for k in group]
                if (len(texts) > 1) and sum(map(len, texts)) <= max_chars:
                    groups.append((tuple(texts), bool(context)))
                    for pos, k in enumerate(group):
                        member[k] = (len(groups) - 1, pos + (1 if context else 0))

            paragraph_chunks = []
            for idx, (voice, part, is_quote) in enumerate(voiced):
                if idx in member:
                    g, pos = member[idx]
                    texts, has_context = groups[g]
                    paragraph_chunks.append(Chunk(part, voice, joint=texts, joint_index=pos, joint_context=has_context))
                elif (prime_replies and not is_quote and idx > 0 and voiced[idx - 1][2] and voiced[idx - 1][0] != voice
                      and part[:1].islower() and len(part) <= max_chars):
                    # A dialogue tag ("said her mother resentfully") continues the quote's sentence. Spoken alone,
                    # TTS models mangle its first word ("said" comes out as "sighed"), so the narrator says the end of
                    # the quote first and that part is cut off again.
                    context = last_clause(voiced[idx - 1][1])
                    paragraph_chunks.append(Chunk(part, voice, joint=(context, part), joint_index=1, joint_context=True))
                else:
                    pieces = [part] if len(part) <= max_chars else pack_sentences(split_sentences(part), max_chars)
                    paragraph_chunks += [Chunk(p, voice) for p in pieces]
                if is_quote and voice != self.narrator_voice:
                    last_line = (voice, part)
            if paragraph_chunks:
                paragraph_chunks[-1].paragraph_end = True
                paragraph_chunks[-1].pause_after = DIALOGUE_PARAGRAPH_PAUSE if has_dialogue else PARAGRAPH_PAUSE
            chunks += paragraph_chunks
        return chunks


def last_clause(text, max_chars=80):
    """The end of a quote, keeping its final punctuation (“My dear Mr. Bennet,” -> 'My dear Mr. Bennet,')."""
    text = text.strip(' “”"‘’')
    sentences = [s for s in re.split(r'(?<=[.!?…])\s+', text) if s]
    last = sentences[-1] if sentences else text
    if len(last) > max_chars:
        last = last[-max_chars:].split(' ', 1)[-1]
    return last


def last_sentence(text, max_chars=120):
    text = text.strip(' “”"‘’')
    sentences = [s for s in re.split(r'(?<=[.!?…])\s+', text) if s]
    last = sentences[-1] if sentences else text
    if len(last) > max_chars:
        last = last[-max_chars:].split(' ', 1)[-1]
    return last if last.endswith(('.', '!', '?', '…')) else last + '.'


def prepare_cast(chapters, book, engine, narrator_voice, llm, output_folder, stem, auto_narrator=False, log=print,
                 llm_casting=True):
    """Loads or builds the cast for the given chapters. Returns (cast, is_new): is_new means the cast sheet was just
    written and should be reviewed before synthesis."""
    cast_path = Path(output_folder) / f'{stem}.cast.toml'
    cache_path = Path(output_folder) / f'{stem}.speakers.json'
    voices = engine.list_voices()
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    speakers_by_chapter = cache.get('chapters', {})
    style = cache.get('quote_style') or detect_quote_style('\n'.join(c.extracted_text for c in chapters[:5]))

    is_new = not cast_path.exists()
    narrator_note = ''
    cached_characters = [Character(c['name'], c.get('gender', 'unknown'), c.get('age', 'adult'), set(c.get('aliases', [])),
                                   description=c.get('description', ''))
                         for c in cache.get('characters', [])]
    if is_new:
        registry = Registry(cached_characters)
        if auto_narrator:
            narrator_voice, narrator_note = auto_pick_narrator(llm, book, chapters, engine, narrator_voice, log, with_note=True)
    else:
        sheet_narrator, characters = read_cast_sheet(cast_path)
        narrator_voice = sheet_narrator or narrator_voice
        registry = Registry(characters)
        for c in cached_characters:  # detected after the cast sheet was written
            if registry.resolve(c.name) in (None, NARRATOR, UNKNOWN):
                registry.add(c.name, c.gender, c.age, c.aliases, c.description)
        log(f'Using cast sheet {cast_path}')

    todo = [c for c in chapters if text_key(c.extracted_text) not in speakers_by_chapter]
    for i, chapter in enumerate(todo, start=1):
        spans = paragraph_spans(chapter.extracted_text, style)
        if not any(s.is_quote for p in spans for s in p):
            speakers_by_chapter[text_key(chapter.extracted_text)] = []
            continue
        log(f'Detecting speakers in "{chapter.get_name()}" ({i}/{len(todo)})...')
        speakers_by_chapter[text_key(chapter.extracted_text)] = attribute_chapter(llm, spans, registry, log)
        characters = [{'name': c.name, 'gender': c.gender, 'age': c.age, 'description': c.description, 'aliases': sorted(c.aliases)}
                      for c in registry.characters.values()]
        cache_path.write_text(json.dumps({'quote_style': style, 'model': llm.model, 'characters': characters,
                                          'chapters': speakers_by_chapter}, ensure_ascii=False, indent=1), encoding='utf-8')

    # Every speaker needs a character entry, even if the LLM forgot to describe them
    for key in (text_key(c.extracted_text) for c in chapters):
        for speaker in speakers_by_chapter.get(key, []):
            if registry.resolve(speaker) is None:
                registry.add(speaker)

    counts = Counter(registry.resolve(s) or s
                     for key in (text_key(c.extracted_text) for c in chapters) for s in speakers_by_chapter.get(key, []))
    for c in registry.characters.values():
        c.lines = max(c.lines, counts.get(c.name, 0))
    speaking = [c for c in registry.characters.values() if c.lines > 0 or c.voice]  # merely mentioned: no voice yet
    new_characters = [c for c in speaking if not c.voice]
    if new_characters:
        if llm is not None and llm_casting:
            log(f'Casting {len(new_characters)} characters with {llm.model}...')
            llm_cast(llm, speaking, voices, narrator_voice, engine.name, book, log)
        accent = voices[narrator_voice].accent if narrator_voice in voices else None
        assign_voices(speaking, voices, narrator_voice, engine.name, accent if accent != 'unknown' else None)
        if not is_new:
            log(f'Added {len(new_characters)} new characters to the cast sheet: {", ".join(c.name for c in new_characters)}')
    if is_new or new_characters:
        write_cast_sheet(cast_path, book.title, book.creator, narrator_voice, speaking, narrator_note, voices)
    return Cast(cast_path, cache_path, narrator_voice, registry, style, speakers_by_chapter), is_new


def print_cast(cast, voices):
    from tabulate import tabulate
    rows = [['(narrator)', '', '', cast.narrator_voice, accent_of(voices, cast.narrator_voice)]]
    rows += [[c.name, c.gender, c.lines, c.voice, accent_of(voices, c.voice)]
             for c in sorted(cast.registry.characters.values(), key=lambda c: -c.lines) if c.voice]
    print(tabulate(rows, headers=['Character', 'Gender', 'Lines', 'Voice', 'Voice accent']))


def accent_of(voices, name):
    v = voices.get(name)
    return v.accent if v and v.accent != 'unknown' else ''
