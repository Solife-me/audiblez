# -*- coding: utf-8 -*-
"""Text-to-speech engines. Every engine turns a chunk of text into mono float32 audio at SAMPLE_RATE."""
import io
import time
from dataclasses import dataclass

import numpy as np

from audiblez.config import api_key, provider_settings

SAMPLE_RATE = 24000

KOKORO_ACCENTS = {'a': 'american', 'b': 'british', 'e': 'spanish', 'f': 'french', 'h': 'hindi', 'i': 'italian',
                  'j': 'japanese', 'p': 'portuguese', 'z': 'chinese'}


@dataclass
class VoiceInfo:
    name: str
    gender: str = 'unknown'  # male | female | unknown
    accent: str = 'unknown'  # american | british | ...
    age: str = 'adult'  # child | young | adult | middle-aged | elderly
    style: str = ''  # 'character' for creature/novelty voices that shouldn't be cast as ordinary people
    expressiveness: float = 0.0  # pitch variation in semitones (from --profile-voices); livelier voices go to main characters
    age_years: float = 0.0  # apparent age measured by --profile-voices
    accent_confidence: float = 1.0  # share of the profiling clips the accent model agreed on
    pitch: float = 0.0  # median pitch in Hz

    @classmethod
    def from_config(cls, name, meta):
        fields = cls.__dataclass_fields__
        return cls(name, **{k: v for k, v in (meta or {}).items() if k in fields and k != 'name' and v not in ('', None)})


class TTSEngine:
    name = 'tts'
    max_chars = 400  # longest chunk of text sent to the model at once
    supports_speed = True
    chars_per_sec = 50  # rough throughput, only used for the ETA

    def synthesize(self, text, voice, speed=1.0):
        raise NotImplementedError

    def list_voices(self):
        """Returns {voice_name: VoiceInfo}."""
        raise NotImplementedError

    def check_voice(self, voice):
        voices = self.list_voices()
        if voices and voice not in voices:
            raise ValueError(f'Voice "{voice}" not available on {self.name}. Available: {", ".join(voices)}')

    def preflight(self, voice, speed=1.0):
        """Synthesizes a word before a long run, so a wrong URL, model or voice fails fast."""
        self.synthesize('Hello.', voice, speed)


class KokoroEngine(TTSEngine):
    name = 'kokoro'
    max_chars = 400  # Kokoro truncates longer inputs for non-english languages

    def __init__(self, pipeline=None):
        self.pipelines = {}
        if pipeline is not None:
            self.pipelines[pipeline.lang_code] = pipeline
        import torch.cuda
        self.chars_per_sec = 500 if torch.cuda.is_available() else 50

    def pipeline(self, lang_code):
        if lang_code not in self.pipelines:
            from kokoro import KPipeline
            from audiblez.core import set_espeak_library
            set_espeak_library()
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
        return self.pipelines[lang_code]

    def synthesize(self, text, voice, speed=1.0):
        pipeline = self.pipeline(voice[0])  # a for american or b for british etc.
        parts = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r'\n\n\n')]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])

    def list_voices(self):
        from audiblez.voices import voices
        return {v: VoiceInfo(v, gender={'f': 'female', 'm': 'male'}.get(v[1], 'unknown'), accent=KOKORO_ACCENTS.get(v[0], 'unknown'))
                for lang in voices.values() for v in lang}


class OpenAICompatibleEngine(TTSEngine):
    """Any server implementing POST {base_url}/audio/speech, e.g. Ollama, OpenAI, Kokoro-FastAPI, LocalAI."""
    chars_per_sec = 300

    def __init__(self, base_url, model, key='', name='openai', max_chars=1000, voices_meta=None, timeout=600, retries=3):
        import requests
        self.session = requests.Session()
        if key:
            self.session.headers['Authorization'] = f'Bearer {key}'
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.name = name
        self.max_chars = max_chars
        self.voices_meta = voices_meta or {}
        self.timeout = timeout
        self.retries = retries
        self.supports_speed = True  # until the server tells us otherwise
        self._voices = None

    def synthesize(self, text, voice, speed=1.0):
        import soundfile
        payload = dict(model=self.model, input=text, voice=voice, response_format='wav')
        if speed != 1.0 and self.supports_speed:
            payload['speed'] = speed
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.post(f'{self.base_url}/audio/speech', json=payload, timeout=self.timeout)
            except Exception as e:  # connection reset, timeout, server restarting...
                if attempt == self.retries: raise
                print(f'TTS request failed ({e}), retrying in {2 ** attempt}s...')
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400 and 'speed' in payload and 'speed' in r.text:
                print(f'{self.name} does not support speed={speed}; speed will be applied when creating the m4b instead.')
                self.supports_speed = False
                payload.pop('speed')
                continue
            if r.status_code >= 500 and attempt < self.retries:
                print(f'TTS server error {r.status_code}, retrying in {2 ** attempt}s...')
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                raise RuntimeError(f'TTS request to {self.base_url} failed with HTTP {r.status_code}: {r.text[:500]}')
            audio, sr = soundfile.read(io.BytesIO(r.content), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return resample(audio, sr, SAMPLE_RATE)
        raise RuntimeError(f'TTS request to {self.base_url} failed after {self.retries} attempts')

    def list_voices(self):
        if self._voices is None:
            names = self._fetch_voice_names()
            names += [n for n in self.voices_meta if n not in names]
            self._voices = {n: VoiceInfo.from_config(n, self.voices_meta.get(n)) for n in names}
        return self._voices

    def _fetch_voice_names(self):
        # Ollama lists a speech model's voices in /api/show; Kokoro-FastAPI and others expose /audio/voices
        root = self.base_url.removesuffix('/v1')
        for method, url, body in [('post', f'{root}/api/show', {'model': self.model}),
                                  ('get', f'{self.base_url}/audio/voices', None)]:
            try:
                r = self.session.request(method, url, json=body, timeout=30)
                if r.status_code != 200: continue
                data = r.json()
                voices = data.get('voices') if isinstance(data, dict) else data
                if voices:
                    return [v if isinstance(v, str) else v.get('id') or v.get('name') for v in voices]
            except Exception:
                continue
        return []


def resample(audio, from_rate, to_rate):
    if from_rate == to_rate or len(audio) == 0:
        return audio
    n = int(round(len(audio) * to_rate / from_rate))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def create_engine(cfg, provider=None, model=None, base_url=None):
    """Builds the TTS engine from the config, with optional per-run overrides."""
    provider = provider or cfg['tts'].get('provider', 'kokoro')
    if provider == 'kokoro' and 'kokoro' not in cfg['providers']:
        cfg['providers']['kokoro'] = {'type': 'kokoro'}
    if base_url and provider not in cfg['providers']:
        cfg['providers'][provider] = {'type': 'openai', 'base_url': base_url}
    settings = provider_settings(cfg, provider)
    if settings['type'] == 'kokoro':
        return KokoroEngine()
    if settings['type'] != 'openai':
        raise ValueError(f'Unknown provider type "{settings["type"]}" for provider "{provider}" (use "kokoro" or "openai")')
    default_model = cfg['tts'].get('model') if provider == cfg['tts'].get('provider') else None
    model = model or default_model or settings.get('tts_model')
    if not model:
        raise ValueError(f'No TTS model set for provider "{provider}". Use --model or set [tts] model in the config.')
    return OpenAICompatibleEngine(
        base_url=base_url or settings['base_url'], model=model, key=api_key(settings), name=provider,
        max_chars=int(settings.get('max_chars', 1000)), voices_meta=settings.get('voices', {}))
