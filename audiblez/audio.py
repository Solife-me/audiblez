# -*- coding: utf-8 -*-
"""Post-processing of TTS clips so they join into continuous speech:
trimming the silence models pad clips with, cutting a joint utterance at its pauses,
and levelling the loudness of different voices."""
import numpy as np

from audiblez.tts import SAMPLE_RATE

FRAME = int(0.01 * SAMPLE_RATE)  # 10 ms analysis frames
TARGET_RMS = 0.112  # about -19 dBFS over voiced frames
PEAK_LIMIT = 0.97


def envelope(audio):
    n = len(audio) // FRAME
    if n == 0:
        return np.zeros(0)
    frames = audio[:n * FRAME].reshape(n, FRAME)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def voiced_threshold(env):
    loud = env[env > 0.005]
    return max(0.006, 0.1 * float(np.median(loud))) if len(loud) else 0.006


def trim(audio, margin=0.03):
    """Removes leading/trailing silence, keeping a small margin, with short fades so cuts never click."""
    env = envelope(audio)
    if len(env) == 0:
        return audio
    voiced = np.nonzero(env > voiced_threshold(env))[0]
    if len(voiced) == 0:
        return audio[:0]
    pad = int(margin * SAMPLE_RATE)
    start = max(0, voiced[0] * FRAME - pad)
    end = min(len(audio), (voiced[-1] + 1) * FRAME + pad)
    return fade(audio[start:end])


def fade(audio, seconds=0.008):
    n = min(int(seconds * SAMPLE_RATE), len(audio) // 2)
    if n:
        audio = audio.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def split_at_pauses(audio, weights, min_pause=0.05, tolerance=0.12):
    """Cuts one synthesized utterance into len(weights) pieces at the pauses closest to where each piece should end,
    judging by its share of the weights (best: the duration of each part spoken on its own). Returns None when a clear pause can't be found near every expected cut,
    so the caller can fall back to synthesizing the pieces separately."""
    env = envelope(audio)
    if len(env) == 0 or len(weights) < 2:
        return None
    quiet = env < voiced_threshold(env)
    runs, start = [], None  # (first_frame, last_frame) of each quiet run inside the utterance
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i - 1))
            start = None
    voiced = np.nonzero(~quiet)[0]
    if len(voiced) == 0:
        return None
    first, last = voiced[0], voiced[-1]
    runs = [(a, b) for a, b in runs if a > first and b < last and (b - a + 1) * 0.01 >= min_pause]
    span = last - first
    total = float(sum(weights))
    cuts, prev, acc = [], first, 0.0
    for w in weights[:-1]:
        acc += w
        expected = first + span * acc / total
        candidates = [(a, b) for a, b in runs if (a + b) / 2 > prev and abs((a + b) / 2 - expected) <= tolerance * span]
        if not candidates:
            return None
        # prefer the longest pause, unless a nearer one is almost as long
        a, b = max(candidates, key=lambda r: (r[1] - r[0]) - 0.3 * abs((r[0] + r[1]) / 2 - expected))
        cut = (a + b) // 2
        cuts.append(cut * FRAME)
        prev = b
    # Sanity check: every piece must take roughly its share of the time, or a cut is in the wrong place
    frame_bounds = [first] + [c // FRAME for c in cuts] + [last]
    for i, w in enumerate(weights):
        expected = span * w / total
        actual = frame_bounds[i + 1] - frame_bounds[i]
        if not 0.7 <= actual / max(expected, 1) <= 1.45:
            return None
    bounds = [0] + cuts + [len(audio)]
    return [audio[bounds[i]:bounds[i + 1]] for i in range(len(weights))]


def longest_pause(audio):
    """Longest silence (seconds) inside a trimmed clip."""
    env = envelope(audio)
    if len(env) == 0:
        return 0.0
    longest = run = 0
    for q in env < voiced_threshold(env):
        run = run + 1 if q else 0
        longest = max(longest, run)
    return longest * FRAME / SAMPLE_RATE


class GlitchDetector:
    """Spots clips where the TTS model went wrong: a long silence in the middle of speech, or a clip far too short or
    long for its text compared with how fast that voice usually speaks (dropped or repeated words)."""
    MAX_PAUSE = 1.6

    def __init__(self):
        self.rates = {}  # voice -> (chars, seconds) of clips that passed

    def problem(self, audio, text, voice):
        seconds = len(audio) / SAMPLE_RATE
        if seconds == 0:
            return 'no audio'
        pause = longest_pause(audio)
        if pause > self.MAX_PAUSE:
            return f'{pause:.1f}s silence inside the clip'
        chars, total = self.rates.get(voice, (0, 0.0))
        if chars >= 300 and len(text) >= 40:
            expected = len(text) * total / chars
            if not 0.6 <= seconds / expected <= 1.7:
                return f'{seconds:.1f}s of audio for text that usually takes {expected:.1f}s'
        return None

    def accept(self, audio, text, voice):
        chars, total = self.rates.get(voice, (0, 0.0))
        self.rates[voice] = (chars + len(text), total + len(audio) / SAMPLE_RATE)


class Leveler:
    """Brings every clip to the same loudness (measured over its voiced frames), so neighbouring clips, whether a
    different voice or a short dialogue tag, never jump in volume. The rise and fall within a clip is kept."""

    def __init__(self, target_rms=TARGET_RMS):
        self.target = target_rms

    def __call__(self, audio, voice=None):
        env = envelope(audio)
        loud = env[env > voiced_threshold(env)] if len(env) else env
        if len(loud) == 0:
            return audio
        gain = self.target / max(float(np.sqrt(np.mean(loud ** 2))), 1e-4)
        peak = float(np.max(np.abs(audio)))
        if peak * gain > PEAK_LIMIT:
            gain = PEAK_LIMIT / peak
        return (audio * gain).astype(np.float32)
