#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import os
import traceback
from glob import glob

import ebooklib
import soundfile
import numpy as np
import time
import shutil
import subprocess
import platform
import re
from types import SimpleNamespace
from tabulate import tabulate
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from audiblez.text import chunk_text, pack_sentences, split_sentences, pause_after_text
from audiblez.audio import Leveler, GlitchDetector, trim, silence, split_at_pauses, longest_pause
from audiblez.tts import TTSEngine, KokoroEngine, SAMPLE_RATE
from ebooklib import epub
from pick import pick

sample_rate = SAMPLE_RATE


def load_spacy():
    """Kept for compatibility: sentence splitting now uses spaCy's rule-based sentencizer, which needs no model download."""


def set_espeak_library():
    """Find the espeak library path"""
    try:

        if os.environ.get('ESPEAK_LIBRARY'):
            library = os.environ['ESPEAK_LIBRARY']
        elif platform.system() == 'Darwin':
            from subprocess import check_output
            try:
                cellar = Path(check_output(["brew", "--cellar"], text=True).strip())
                pattern = cellar / "espeak-ng" / "*" / "lib" / "*.dylib"
                if not (library := next(iter(glob(str(pattern))), None)):
                    raise RuntimeError("No espeak-ng library found; please set the path manually")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError("Cannot locate Homebrew Cellar. Is 'brew' installed and in PATH?") from e
        elif platform.system() == 'Linux':
            library = glob('/usr/lib/*/libespeak-ng*')[0]
        elif platform.system() == 'Windows':
            library = 'C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll'
        else:
            print('Unsupported OS, please set the espeak library path manually')
            return
        print('Using espeak library:', library)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(library)
    except Exception:
        traceback.print_exc()
        print("Error finding espeak-ng library:")
        print("Probably you haven't installed espeak-ng.")
        print("On Mac: brew install espeak-ng")
        print("On Linux: sudo apt install espeak-ng")


def main(file_path, voice=None, pick_manually=False, speed=1.0, output_folder='.',
         max_chapters=None, max_sentences=None, selected_chapters=None, post_event=None, engine=None,
         characters=False, llm=None, review=True, auto_narrator=False, prime_replies=False, verifier=None,
         llm_casting=True, skip_footnotes=True, roman_numerals=True):
    if post_event: post_event('CORE_STARTED')
    load_spacy()
    if output_folder != '.':
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    stem = Path(file_path).stem
    engine = engine or KokoroEngine()
    voice = voice or 'af_sky'
    speed = speed or 1.0
    engine.check_voice(voice)

    book = load_book(file_path, skip_footnotes=skip_footnotes)
    title, creator, document_chapters = book.title, book.creator, book.chapters

    if not selected_chapters:
        if pick_manually is True:
            selected_chapters = pick_chapters(document_chapters)
        else:
            selected_chapters = book.good_chapters
    print_selected_chapters(document_chapters, selected_chapters)
    if max_chapters:
        selected_chapters = selected_chapters[:max_chapters]
    texts = [c.extracted_text for c in selected_chapters]

    cast = None
    if (characters or auto_narrator) and llm is None:
        raise ValueError('Character voices and --auto-narrator need an LLM: set [llm] in the config or pass --llm-model.')
    if characters:
        from audiblez.characters import prepare_cast, print_cast
        cast, is_new = prepare_cast(selected_chapters, book, engine, voice, llm, output_folder, stem, auto_narrator=auto_narrator,
                                    llm_casting=llm_casting)
        voice = cast.narrator_voice
        print_cast(cast, engine.list_voices())
        for v in {c.voice for c in cast.registry.characters.values() if c.voice} | {voice}:
            engine.check_voice(v)
        if is_new and review:
            print(f'\nReview the cast in {cast.cast_path}, then run the same command again to create the audiobook.')
            return
    elif auto_narrator:
        from audiblez.characters import auto_pick_narrator
        voice = auto_pick_narrator(llm, book, selected_chapters, engine, voice)

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    stats = SimpleNamespace(
        total_chars=sum(map(len, texts)),
        processed_chars=0,
        chars_per_sec=engine.chars_per_sec)
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Using {engine.name} with voice {voice}')
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')
    engine.preflight(voice, speed)
    if verifier is None and engine.name != 'kokoro' and engine.max_chars > UNVERIFIED_MAX_CHARS:
        # Nothing will catch a TTS model looping on a long input, so read in pieces short enough not to trigger it
        engine.max_chars = UNVERIFIED_MAX_CHARS

    chapter_wav_files = []
    chapter_titles = []
    for i, chapter in enumerate(selected_chapters, start=1):
        text = chapter.extracted_text
        voice_tag = slug(voice) + (f'_cast{cast.signature()}{"p" if prime_replies else ""}' if cast else '')
        voice_tag += ('_withnotes' if not skip_footnotes else '') + ('_roman' if not roman_numerals else '')
        chapter_wav_path = Path(output_folder) / f'{stem}_chapter_{i}_{voice_tag}_{slug(chapter.get_name())}.wav'
        if Path(chapter_wav_path).exists():
            print(f'File for chapter {i} already exists. Skipping')
            stats.processed_chars += len(text)
            chapter_wav_files.append(chapter_wav_path)
            chapter_titles.append(chapter_title(chapter, i))
            if post_event:
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            continue
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            continue
        intro = f'{title} – {creator}.' if i == 1 else ''  # add intro text
        start_time = time.time()
        if post_event: post_event('CORE_CHAPTER_STARTED', chapter_index=chapter.chapter_index)
        if cast:
            chunks = (chunk_text(intro, voice, engine.max_chars, roman_numerals)
                      + cast.chunks(chapter.extracted_text, engine.max_chars, prime_replies, roman_numerals))
        else:
            chunks = chunk_text(f'{intro}\n\n{text}' if intro else text, voice, engine.max_chars, roman_numerals)
        if synthesize_chunks(engine, chunks, speed, chapter_wav_path, stats, post_event=post_event, max_chunks=max_sentences,
                             verifier=verifier):
            chapter_wav_files.append(chapter_wav_path)
            chapter_titles.append(chapter_title(chapter, i))
            delta_seconds = time.time() - start_time
            print('Chapter written to', chapter_wav_path)
            if post_event: post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({len(text) / delta_seconds:.0f} characters per second)')
        else:
            print(f'Warning: No audio generated for chapter {i}')

    if has_ffmpeg and chapter_wav_files:
        tempo = 1.0 if engine.supports_speed else speed
        create_index_file(title, creator, chapter_wav_files, output_folder, chapter_titles, tempo=tempo)
        create_m4b(chapter_wav_files, f'{stem}.m4b', book.cover_image, output_folder, tempo=tempo)
        if post_event: post_event('CORE_FINISHED')


def slug(name):
    return re.sub(r'[^\w.-]+', '_', name).strip('_')


def chapter_title(chapter, i):
    return getattr(chapter, 'title', None) or f'Chapter {i}'


def load_book(file_path, skip_footnotes=True):
    """Returns a SimpleNamespace(title, creator, cover_image, chapters, good_chapters) for an epub or pdf."""
    if Path(file_path).suffix.lower() == '.pdf':
        from audiblez.pdf import load_pdf
        return load_pdf(file_path, skip_footnotes=skip_footnotes)
    book = epub.read_epub(file_path)
    meta_title = book.get_metadata('DC', 'title')
    meta_creator = book.get_metadata('DC', 'creator')
    cover_maybe = find_cover(book)
    if cover_maybe:
        print(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')
    document_chapters = find_document_chapters_and_extract_texts(book, skip_footnotes=skip_footnotes)
    return SimpleNamespace(
        title=meta_title[0][0] if meta_title else '',
        creator=meta_creator[0][0] if meta_creator else '',
        cover_image=cover_maybe.get_content() if cover_maybe else b'',
        chapters=document_chapters,
        good_chapters=find_good_chapters(document_chapters))


def find_cover(book):
    def is_image(item):
        return item is not None and item.media_type.startswith('image/')

    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        if is_image(item):
            return item

    # https://idpf.org/forum/topic-715
    for meta in book.get_metadata('OPF', 'cover'):
        if is_image(item := book.get_item_with_id(meta[1]['content'])):
            return item

    if is_image(item := book.get_item_with_id('cover')):
        return item

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower() and is_image(item):
            return item

    return None


def print_selected_chapters(document_chapters, chapters):
    ok = 'X' if platform.system() == 'Windows' else '✅'
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if c in chapters else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))

def synthesize_chunks(engine, chunks, speed, out_path, stats=None, post_event=None, max_chunks=None, verifier=None):
    """Synthesizes the chunks straight into a wav file, so memory use doesn't grow with the chapter length.
    Each clip is trimmed of the model's padding silence, levelled to the same loudness per voice and followed by
    a pause that fits its punctuation. Writes to a .part file first so an interrupted chapter is redone on the next
    run. Returns False if no audio."""
    out_path = Path(out_path)
    tmp_path = out_path.with_name(out_path.name + '.part')
    leveler = Leveler()
    glitches = GlitchDetector(verifier)
    joint_pieces = {}  # joint texts -> trimmed audio pieces, for chunks synthesized as one utterance
    samples_written = 0
    with soundfile.SoundFile(tmp_path, 'w', samplerate=sample_rate, channels=1, subtype='PCM_16', format='WAV') as f:
        for i, chunk in enumerate(chunks):
            if max_chunks and i > max_chunks: break
            started = time.time()
            if chunk.joint:
                key = (chunk.voice, chunk.joint)
                if key not in joint_pieces:
                    joint_pieces[key] = synthesize_joint(engine, chunk, speed, glitches)
                audio = joint_pieces[key][chunk.joint_index]
            else:
                audio = synthesize_clip(engine, chunk.text, chunk.voice, speed, glitches)
            if len(audio) == 0:
                continue
            audio = leveler(audio, chunk.voice)
            f.write(audio)
            f.write(silence(chunk.pause()))
            samples_written += len(audio)
            if stats:
                stats.processed_chars += len(chunk.text)
                measured = len(chunk.text) / max(time.time() - started, 1e-3)
                stats.chars_per_sec = 0.9 * stats.chars_per_sec + 0.1 * measured
                stats.progress = min(100, stats.processed_chars * 100 // max(stats.total_chars, 1))
                stats.eta = strfdelta(max(stats.total_chars - stats.processed_chars, 0) / stats.chars_per_sec)
                if post_event: post_event('CORE_PROGRESS', stats=stats)
                print(f'Estimated time remaining: {stats.eta}')
                print('Progress:', f'{stats.progress}%\n')
    if samples_written == 0:
        tmp_path.unlink()
        return False
    tmp_path.replace(out_path)
    return True


SPLIT_RETRY_CHARS = 300  # a clip that keeps failing is re-synthesized in pieces this long
UNVERIFIED_MAX_CHARS = 400  # without verification: in testing, phonon never looped on inputs this short


def synthesize_clip(engine, text, voice, speed, glitches, attempts=2, depth=0):
    """Synthesizes and trims one clip, regenerating it when the TTS output looks broken (a long silence, a wrong
    length or, with a verifier, dropped, repeated or invented words). If it still fails, the text is synthesized in
    shorter pieces: TTS models loop on long inputs ("he loves them without works, he loves them without works"),
    and resending the same text tends to fail the same way. Keeps the best attempt if nothing works."""
    best, best_score, best_problem = None, None, None
    verifier = glitches.verifier if glitches.verifier and glitches.verifier.applies(engine, voice) else None
    for attempt in range(attempts):
        audio = trim(engine.synthesize(text, voice, speed))
        problem, score = glitches.problem(audio, text, voice), 0
        if problem:
            score = 1000 + longest_pause(audio)
        elif verifier:
            wrong = verifier.problem(audio, text)
            if wrong:
                score, problem = wrong
        if problem is None:
            glitches.accept(audio, text, voice)
            return audio
        print(f'TTS glitch ({problem}) in "{text[:50]}...", regenerating ({attempt + 1}/{attempts})')
        if best is None or score < best_score:
            best, best_score, best_problem = audio, score, problem

    if depth < 2 and len(text) > 60:
        limit = SPLIT_RETRY_CHARS if len(text) > SPLIT_RETRY_CHARS * 1.2 else max(60, len(text) // 2)
        pieces = pack_sentences(split_sentences(text), limit)
        if len(pieces) > 1:
            print(f'Synthesizing it in {len(pieces)} shorter pieces instead')
            parts = []
            for i, piece in enumerate(pieces):
                parts.append(synthesize_clip(engine, piece, voice, speed, glitches, attempts, depth + 1))
                if i < len(pieces) - 1:
                    parts.append(silence(pause_after_text(piece)))
            return np.concatenate(parts)
    print(f'Warning: keeping a clip that still has a problem ({best_problem}): "{text[:80]}..."')
    return best


def synthesize_joint(engine, chunk, speed, glitches=None):
    """Synthesizes all the texts of a joint utterance in one request, so each part keeps the intonation it has in
    context, then cuts it at its pauses. Each part is also synthesized on its own: its length tells where the cut
    belongs (far more precisely than a character count), and it is the fallback when no clean cut is found."""
    glitches = glitches or GlitchDetector()
    texts = [t.strip(' “”"‘’') for t in chunk.joint]
    separate = [synthesize_clip(engine, t, chunk.voice, speed, glitches) for t in texts]
    joint = synthesize_clip(engine, ' '.join(texts), chunk.voice, speed, glitches)
    # A context line that ends a sentence is followed by a sentence pause; insist on one there, so the cut can't
    # land in the short gap after the first word of the line that follows (“I | am sick of Mr. Bingley”)
    min_pauses = [0.15 if chunk.joint_context and i == 0 and texts[0].rstrip(' ”"’').endswith(('.', '!', '?', '…')) else 0.05
                  for i in range(len(texts) - 1)]
    pieces = split_at_pauses(joint, [max(len(p), 1) for p in separate], min_pauses=min_pauses)
    verifier = glitches.verifier if glitches.verifier and glitches.verifier.applies(engine, chunk.voice) else None
    if pieces is not None and verifier and any(verifier.problem(piece, text) for i, (piece, text) in enumerate(zip(pieces, texts))
                                               if not (chunk.joint_context and i == 0)):
        pieces = None  # a cut landed in the wrong place
    if pieces is None:
        pieces = separate
    pieces = [trim(p) for p in pieces]
    if chunk.joint_context:
        pieces[0] = None  # the context line is only there to set the tone
    return pieces


def gen_audio_segments(engine, text, voice, speed, stats=None, max_sentences=None, post_event=None):
    """Returns the audio for text as a list of numpy arrays. engine may also be a Kokoro KPipeline."""
    if not isinstance(engine, TTSEngine):
        engine = KokoroEngine(pipeline=engine)
    audio_segments = []
    for i, chunk in enumerate(chunk_text(text, voice, engine.max_chars)):
        if max_sentences and i > max_sentences: break
        audio_segments.append(engine.synthesize(chunk.text, chunk.voice, speed))
        if stats:
            stats.processed_chars += len(chunk.text)
            stats.progress = stats.processed_chars * 100 // stats.total_chars
            stats.eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
            if post_event: post_event('CORE_PROGRESS', stats=stats)
    return audio_segments


def gen_text(text, voice='af_heart', output_file='text.wav', speed=1, play=False, engine=None):
    load_spacy()
    audio_segments = gen_audio_segments(engine or KokoroEngine(), text, voice=voice, speed=speed)
    final_audio = np.concatenate(audio_segments)
    soundfile.write(output_file, final_audio, sample_rate)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])


FOOTNOTE_TYPES = re.compile(r'\b(footnotes?|endnotes?|rearnotes?|notes?)\b')
FOOTNOTE_ROLES = {'doc-footnote', 'doc-endnote', 'doc-endnotes', 'doc-footnotes'}
FOOTNOTE_CLASS = re.compile(r'^(foot|fn|fnote|ftn)$|footnote|endnote', re.I)
NOTE_LINK = re.compile(r'note|fn|ftn|foot', re.I)
MARKER_TEXT = re.compile(r'^[\[(]?\s*(\d{1,4}[a-z]?|[a-z]|[*†‡§¶]+)\s*[\])]?$', re.I)


def strip_footnotes(soup):
    """Removes footnote/endnote bodies and the reference markers in the text (which would be read as "twelve")."""
    def is_note(el):
        types = el.get('epub:type', '') or ''
        classes = el.get('class') or []
        return ((FOOTNOTE_TYPES.search(types) and 'noteref' not in types) or el.get('role') in FOOTNOTE_ROLES
                or any(FOOTNOTE_CLASS.search(c) for c in classes))

    def is_marker(el):
        text = el.get_text(strip=True)
        if not MARKER_TEXT.match(text or '-'):
            return False
        if el.name == 'a':
            return ('noteref' in (el.get('epub:type', '') or '') or el.get('role') == 'doc-noteref'
                    or bool(NOTE_LINK.search((el.get('href') or '') + ' ' + (el.get('id') or ''))))
        return el.name == 'sup' and (el.find('a') is not None or not text.isalpha())  # not the "th" of 19<sup>th</sup>

    for el in soup.find_all(is_note):
        el.decompose()
    for el in soup.find_all(['a', 'sup']):
        if not el.decomposed and is_marker(el):
            parent = el.parent
            el.decompose()
            if parent is not None and parent.name == 'sup' and not parent.get_text(strip=True):
                parent.decompose()


CONTENT_TAGS = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'li']


def own_text(element):
    """The element's text, minus the text of nested content elements (a <p> inside an <li>), which are read on their
    own. Otherwise every paragraph wrapped in a list item is read twice in a row."""
    if not element.find(CONTENT_TAGS):
        return element.get_text().strip()
    parts = []
    for string in element.find_all(string=True):
        owner = next((parent for parent in string.parents if parent.name in CONTENT_TAGS), None)
        if owner is element:
            parts.append(string)
    return ''.join(parts).strip()


def find_document_chapters_and_extract_texts(book, skip_footnotes=True):
    """Returns every chapter that is an ITEM_DOCUMENT and enriches each chapter with extracted_text."""
    document_chapters = []
    for chapter in book.get_items():
        if chapter.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        if skip_footnotes:
            strip_footnotes(soup)
        chapter.extracted_text = ''
        for text in [own_text(c) for c in soup.find_all(CONTENT_TAGS)]:
            text = re.sub(r'\s+', ' ', text)  # line breaks inside a paragraph are just source formatting
            text = re.sub(r'\s*\[\s*\]|\(\s*\)', '', text).strip()  # brackets left around removed note markers
            if not text:
                continue
            if not re.search(r'[.!?…:;”"’)\]]$', text):
                text += '.'
            chapter.extracted_text += text + '\n'
        document_chapters.append(chapter)
    for i, c in enumerate(document_chapters):
        c.chapter_index = i  # this is used in the UI to identify chapters
    return document_chapters


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name.lower()
        or re.search(r'part_?\d{1,3}', name)
        or re.search(r'split_?\d{1,3}', name)
        or re.search(r'ch_?\d{1,3}', name)
        or re.search(r'chap_?\d{1,3}', name)
    )
    return has_min_len and title_looks_like_chapter


def chapter_beginning_one_liner(c, chars=20):
    s = c.extracted_text[:chars].strip().replace('\n', ' ').replace('\r', ' ')
    return s + '…' if len(s) > 0 else ''


NAVIGATION_DOCUMENT = re.compile(r'(^|/)(toc|nav|contents?|titlepage|cover)\.x?html?$', re.I)


def find_good_chapters(document_chapters):
    chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if len(chapters) == 0:
        print('Not easy to recognize the chapters, defaulting to all non-empty documents.')
        chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and len(c.extracted_text) > 10
                    and not NAVIGATION_DOCUMENT.search(c.get_name())]
    return chapters


def pick_chapters(chapters):
    # Display the document name, the length and first 50 characters of the text
    chapters_by_names = {
        f'{c.get_name()}\t({len(c.extracted_text)} chars)\t[{chapter_beginning_one_liner(c, 50)}]': c
        for c in chapters}
    title = 'Select which chapters to read in the audiobook'
    ret = pick(list(chapters_by_names.keys()), title, multiselect=True, min_selection_count=1)
    selected_chapters_out_of_order = [chapters_by_names[r[0]] for r in ret]
    selected_chapters = [c for c in chapters if c in selected_chapters_out_of_order]
    return selected_chapters


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def create_m4b(chapter_files, filename, cover_image, output_folder, tempo=1.0):
    """Encode the chapter wavs into a single m4b with chapter markers and (optionally) the cover image, in one ffmpeg pass."""
    stem = Path(filename).stem
    wav_list_txt = Path(output_folder) / f'{stem}_wav_list.txt'
    with open(wav_list_txt, 'w', encoding='utf-8') as f:
        for wav_file in chapter_files:
            escaped = str(Path(wav_file).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    final_filename = Path(output_folder) / f'{stem}.m4b'
    chapters_txt_path = Path(output_folder) / "chapters.txt"
    print('Creating M4B file...')

    if cover_image:
        cover_file_path = Path(output_folder) / 'cover'
        with open(cover_file_path, 'wb') as f:
            f.write(cover_image)
        cover_image_args = [
            '-i', f'{cover_file_path}',
            '-map', '2:v',  # Map cover image
            '-disposition:v', 'attached_pic',  # Ensure cover is embedded
            '-c:v', 'copy',  # Keep cover unchanged
        ]
    else:
        cover_image_args = []

    proc = subprocess.run([
        'ffmpeg',
        '-y',  # Overwrite output
        '-f', 'concat', '-safe', '0', '-i', f'{wav_list_txt}',  # Input audio (all chapter wavs)
        '-i', f'{chapters_txt_path}',  # Input chapters
        *cover_image_args,  # Cover image (if provided)

        '-map', '0:a',  # Map audio
        *(['-filter:a', f'atempo={tempo}'] if tempo != 1.0 else []),  # Speed, for engines that can't change it
        '-c:a', 'aac',  # Built-in AAC encoder, available in every ffmpeg build
        '-b:a', '64k',  # Reduce bitrate for smaller size

        '-map_metadata', '1',  # Map metadata
        '-map_chapters', '1',  # Map chapter markers

        '-f', 'mp4',  # Output as M4B
        f'{final_filename}'  # Output file
    ])

    wav_list_txt.unlink()
    if proc.returncode == 0:
        print(f'{final_filename} created. Enjoy your audiobook.')
        print('Feel free to delete the intermediary .wav chapter files, the .m4b is all you need.')
    else:
        print(f'ffmpeg failed to create {final_filename} (exit code {proc.returncode})')


def probe_duration(file_name):
    args = ['ffprobe', '-i', file_name, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def create_index_file(title, creator, chapter_mp3_files, output_folder, chapter_titles=None, tempo=1.0):
    with open(Path(output_folder) / "chapters.txt", "w", encoding="utf-8") as f:
        f.write(f";FFMETADATA1\ntitle={ffmeta_escape(title)}\nartist={ffmeta_escape(creator)}\n\n")
        start = 0
        for i, c in enumerate(chapter_mp3_files):
            duration = probe_duration(c) / tempo
            end = start + int(duration * 1000)
            name = chapter_titles[i] if chapter_titles else f'Chapter {i}'
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={ffmeta_escape(name)}\n\n")
            start = end


def ffmeta_escape(value):
    """Escapes the characters that have a special meaning in ffmpeg metadata files."""
    return re.sub(r'([=;#\\\n])', r'\\\1', str(value))
