# -*- coding: utf-8 -*-
"""Voice profiling: measures each voice's gender, apparent age, accent, pitch and expressiveness from a few rendered
sentences, so characters can be cast with voices that fit them. Needs `pip install "audiblez[profile]"`.

Models (downloaded from Hugging Face on first use):
  - age and gender: audeering/wav2vec2-large-robust-24-ft-age-gender (CC BY-NC-SA 4.0: non-commercial use)
  - accent: Jzuluaga/accent-id-commonaccent_ecapa (16 English accents, trained on Common Voice)
"""
import collections
import re
from pathlib import Path

import numpy as np

from audiblez.tts import SAMPLE_RATE

PROFILE_LINES = [
    'The morning was cold and grey, and nobody in the village had seen the stranger arrive before the storm.',
    'I have been thinking about what you said yesterday, and I believe you were right all along.',
    'If you follow the river past the old mill, you will find the cottage where my grandmother used to live.',
    'We should leave before it gets dark. The roads are dangerous at night, and the horses are tired.',
]
ACCENTS = {'england': 'british', 'us': 'american', 'scotland': 'scottish', 'ireland': 'irish', 'wales': 'welsh',
           'australia': 'australian', 'canada': 'canadian', 'newzealand': 'new zealand', 'indian': 'indian',
           'african': 'south african', 'malaysia': 'malaysian', 'philippines': 'filipino', 'singapore': 'singaporean',
           'hongkong': 'hong kong', 'bermuda': 'bermudian', 'southatlandtic': 'south atlantic'}
AGE_MODEL = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
ACCENT_MODEL = 'Jzuluaga/accent-id-commonaccent_ecapa'


def age_group(years):
    if years < 13: return 'child'
    if years < 26: return 'young'
    if years < 46: return 'adult'
    if years < 61: return 'middle-aged'
    return 'elderly'


def to16k(audio):
    n = int(len(audio) * 16000 / SAMPLE_RATE)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


class Profiler:
    def __init__(self, cache_dir):
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
        from speechbrain.inference.classifiers import EncoderClassifier

        class Head(nn.Module):
            def __init__(self, config, labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, labels)

            def forward(self, x):
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class AgeGender(Wav2Vec2PreTrainedModel):  # the model card's architecture
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = Head(config, 1)
                self.gender = Head(config, 3)  # female, male, child
                self.post_init()

            def forward(self, x):
                hidden = torch.mean(self.wav2vec2(x)[0], dim=1)
                return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

        print('Loading the age/gender and accent models (downloaded on first use)...')
        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(AGE_MODEL)
        self.age_gender = AgeGender.from_pretrained(AGE_MODEL).eval()
        self.accent = EncoderClassifier.from_hparams(source=ACCENT_MODEL, savedir=str(Path(cache_dir) / 'accent-model'))

    def profile(self, clips, expression_clips):
        from audiblez.audition import median_f0, pitch_variation
        ages, genders, accents = [], [], collections.Counter()
        for clip in clips:
            x = to16k(clip)
            with self.torch.no_grad():
                age, gender = self.age_gender(self.processor(x, sampling_rate=16000, return_tensors='pt').input_values)
                accent = self.accent.classify_batch(self.torch.tensor(x).unsqueeze(0))[3][0]
            ages.append(float(age[0, 0]) * 100)
            genders.append(gender[0].numpy())
            accents[accent] += 1
        female, male, child = np.mean(genders, axis=0)
        accent, votes = accents.most_common(1)[0]
        variation = [v for v in (pitch_variation(c) for c in expression_clips) if v]
        years = float(np.mean(ages))
        return dict(
            gender='female' if female >= male else 'male',
            age=('child' if child > 0.5 else age_group(years)), age_years=round(years),
            accent=ACCENTS.get(accent, accent), accent_confidence=round(votes / len(clips), 2),
            pitch=round(float(np.median([median_f0(c) for c in clips]))),
            expressiveness=round(float(np.mean(variation)), 1) if variation else 0.0)


def profile_voices(engine, cache_dir='.', voices=None, log=print):
    """Returns {voice: profile} for the engine's voices (or the given ones)."""
    from audiblez.audio import trim
    from audiblez.audition import EXPRESSION_LINES
    profiler = Profiler(cache_dir)
    known = engine.list_voices()
    results = {}
    for name in voices or known:
        log(f'Profiling {name}...')
        clips = [trim(engine.synthesize(line, name)) for line in PROFILE_LINES]
        expression = [trim(engine.synthesize(line, name)) for line in EXPRESSION_LINES]
        results[name] = profiler.profile(clips, expression)
        if name in known and getattr(known[name], 'style', ''):
            results[name]['style'] = known[name].style  # keep the user's novelty-voice tag
    return results


def toml_key(name):
    return name if re.fullmatch(r'[A-Za-z0-9_-]+', name) else '"' + name.replace('"', '\\"') + '"'


def toml_value(v):
    return f'"{v}"' if isinstance(v, str) else str(v)


def voices_section(provider, profiles):
    lines = [f'[providers.{provider}.voices]']
    for name, p in profiles.items():
        fields = ', '.join(f'{k} = {toml_value(v)}' for k, v in p.items())
        note = '  # accent uncertain' if p.get('accent_confidence', 1) < 0.75 else ''
        lines.append(f'{toml_key(name)} = {{ {fields} }}{note}')
    return '\n'.join(lines) + '\n'


def update_config(config_path, provider, profiles):
    """Replaces the [providers.<provider>.voices] table in the config with the measured profiles."""
    path = Path(config_path)
    text = path.read_text(encoding='utf-8') if path.exists() else ''
    header = f'[providers.{provider}.voices]'
    section = voices_section(provider, profiles)
    if header in text:
        start = text.index(header)
        nxt = re.search(r'^\[', text[start + len(header):], flags=re.M)
        end = start + len(header) + nxt.start() if nxt else len(text)
        # keep the comment block right above the next table with the next table
        text = text[:start] + section + ('\n' + text[end:] if nxt else '')
    else:
        text = text.rstrip('\n') + '\n\n' + section
    path.write_text(text, encoding='utf-8')
