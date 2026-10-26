# -*- coding: utf-8 -*-
"""Optional check of every TTS clip with a small speech recognizer (faster-whisper), to catch the rare clips where
the TTS model drops part of the text or invents words. Install with `pip install audiblez[verify]`."""
import difflib
import importlib.util
import re

import numpy as np

from audiblez.tts import SAMPLE_RATE

MIN_RUN = 4  # dropped or invented runs shorter than this are usually the recognizer's own mistakes
SPOKEN_FORMS = {'mister': 'mr', 'missus': 'mrs', 'miz': 'ms', 'doctor': 'dr', 'saint': 'st'}


def available():
    return importlib.util.find_spec('faster_whisper') is not None


def words(text):
    text = text.lower().replace('’', "'").replace("'", '')
    return [SPOKEN_FORMS.get(w, w) for w in re.findall(r'[a-z0-9]+', text)]


class Verifier:
    def __init__(self, model='base.en'):
        self.model_name = model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f'Loading speech recognizer {self.model_name} to verify the audio...')
            self._model = WhisperModel(self.model_name, device='cpu', compute_type='int8')
        return self._model

    def applies(self, engine, voice):
        """English-only recognizer models (*.en) can't check Kokoro voices of other languages."""
        return not (self.model_name.endswith('.en') and engine.name == 'kokoro' and voice[:1] not in 'ab')

    def transcribe(self, audio):
        n = int(len(audio) * 16000 / SAMPLE_RATE)
        audio16 = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        segments, _ = self.model.transcribe(audio16, beam_size=1, without_timestamps=True,
                                            condition_on_previous_text=False, vad_filter=False)
        return ' '.join(s.text for s in segments)

    def problem(self, audio, text):
        """Returns (number of wrong words, description) when the clip doesn't say the text, else None."""
        expected = words(text)
        if len(expected) < MIN_RUN:
            return None  # too short to drop or invent a run of words
        heard = words(self.transcribe(audio))
        wrong, details = 0, []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes():
            missing, extra = i2 - i1 if tag in ('delete', 'replace') else 0, j2 - j1 if tag in ('insert', 'replace') else 0
            if missing >= MIN_RUN and extra < missing / 2:
                wrong += missing
                details.append(f'dropped "{" ".join(expected[i1:i2])}"')
            elif extra >= MIN_RUN and missing < extra / 2:
                wrong += extra
                details.append(f'added "{" ".join(heard[j1:j2])}"')
        return (wrong, '; '.join(details)) if wrong else None


def create_verifier(cfg):
    setting = cfg['tts'].get('verify', 'auto')
    if setting is False or setting == 'false':
        return None
    if not available():
        if setting is True or setting == 'true':
            print('Audio verification needs faster-whisper: pip install "audiblez[verify]". Continuing without it.')
        return None
    return Verifier(cfg['tts'].get('verify_model', 'base.en'))
