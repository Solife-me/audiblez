# -*- coding: utf-8 -*-
"""Voice audition: renders a short sample of every voice into one m4b (a chapter per voice), and guesses each
voice's gender from its pitch, so untagged voices (e.g. on Ollama) can be tagged in the config."""
from pathlib import Path

import numpy as np
import soundfile

from audiblez.tts import SAMPLE_RATE

SAMPLE_TEXT = ('Hello, my name is {name}. The morning was cold and grey, and nobody in the village '
               'had seen the stranger arrive. "Where are you going?" she asked.')


def median_f0(audio, sr=SAMPLE_RATE, fmin=60, fmax=400):
    """Median fundamental frequency (Hz) of the voiced frames, via normalized autocorrelation."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    f0s = []
    for start in range(0, len(audio) - frame, hop):
        x = audio[start:start + frame].astype(np.float64)
        if np.sqrt(np.mean(x ** 2)) < 0.02:
            continue  # silence
        x = x - x.mean()
        ac = np.correlate(x, x, 'full')[frame - 1:]
        ac /= ac[0] + 1e-9
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else 0.0


def gender_from_pitch(f0):
    """Adult speech: men mostly 85-155 Hz, women 165-255 Hz. Between them the guess is unreliable."""
    if f0 >= 170: return 'female', True
    if 0 < f0 < 145: return 'male', True
    return ('female' if f0 >= 158 else 'male'), False


def audition(engine, output_folder='.'):
    from audiblez.core import create_index_file, create_m4b
    voices = engine.list_voices()
    if not voices:
        print(f'{engine.name} did not report any voices.')
        return
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    wavs, titles, rows = [], [], []
    for name, info in voices.items():
        print(f'Rendering {name}...')
        audio = engine.synthesize(SAMPLE_TEXT.format(name=name), name)
        wav = Path(output_folder) / f'audition_{engine.name}_{len(wavs):03d}.wav'
        soundfile.write(wav, audio, SAMPLE_RATE)
        wavs.append(wav)
        titles.append(name)
        f0 = median_f0(audio)
        guess, sure = gender_from_pitch(f0)
        rows.append((name, info, f0, guess, sure))
    create_index_file(f'{engine.name} voices', 'audiblez', wavs, output_folder, titles)
    create_m4b(wavs, f'{engine.name}_voices.m4b', b'', output_folder)
    for wav in wavs:
        wav.unlink()

    print(f'\nPaste this into your config and correct it after listening to {Path(output_folder) / (engine.name + "_voices.m4b")}.')
    print('Pitch-based gender guesses marked "check" are uncertain. Fill in the accents, and add style = "character"')
    print('to creature/novelty voices so they are not cast as ordinary people.\n')
    print(f'[providers.{engine.name}.voices]')
    for name, info, f0, guess, sure in rows:
        gender = info.gender if info.gender != 'unknown' else guess
        accent = info.accent if info.accent != 'unknown' else ''
        key = name if name.replace('_', '').replace('-', '').isalnum() else f'"{name}"'
        note = f'  # {f0:.0f} Hz' + ('' if sure or info.gender != 'unknown' else ', check')
        print(f'{key} = {{ gender = "{gender}", accent = "{accent}" }}{note}')
