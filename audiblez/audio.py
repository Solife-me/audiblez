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
    """Level of clearly voiced speech; used to measure loudness."""
    loud = env[env > 0.005]
    return max(0.006, 0.1 * float(np.median(loud))) if len(loud) else 0.006


def silence_threshold(env):
    """Level just above the clip's own noise floor. Anything louder is sound, including the quiet start of
    fricatives like the f in "fortnight" or the h in "he", which a speech-level threshold would cut off."""
    if len(env) == 0:
        return 0.0008
    loud = env[env > 0.005]
    ceiling = 0.3 * float(np.median(loud)) if len(loud) else 0.0008  # for clips with no silence to measure
    return max(min(8 * float(np.percentile(env, 5)), ceiling), 0.0008)


def sounding(env):
    """Frames that are sound, ignoring isolated blips (some models start every clip with a tiny click)."""
    above = env > silence_threshold(env)
    steady = above.copy()
    steady[:-2] = above[:-2] & above[1:-1] & above[2:]
    steady[-2:] = False
    # extend each steady run back over the two frames that started it
    for shift in (1, 2):
        steady[:-shift] |= steady[shift:] & above[:-shift]
    return steady


def trim(audio, lead=0.04, tail=0.06):
    """Removes the silence before and after the speech, keeping a small margin, with short fades so cuts never
    click. Only true silence is removed: soft consonants at the edges are kept."""
    env = envelope(audio)
    if len(env) == 0:
        return audio
    sound = np.nonzero(sounding(env))[0]
    if len(sound) == 0:
        return audio[:0]
    start = max(0, sound[0] * FRAME - int(lead * SAMPLE_RATE))
    end = min(len(audio), (sound[-1] + 1) * FRAME + int(tail * SAMPLE_RATE))
    return fade(audio[start:end])


def fade(audio, seconds=0.004):
    n = min(int(seconds * SAMPLE_RATE), len(audio) // 2)
    if n:
        audio = audio.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def split_at_pauses(audio, weights, min_pause=0.05, tolerance=0.12, min_pauses=None):
    """Cuts one synthesized utterance into len(weights) pieces at the pauses closest to where each piece should end,
    judging by its share of the weights (best: the duration of each part spoken on its own). Returns None when a clear pause can't be found near every expected cut,
    so the caller can fall back to synthesizing the pieces separately. min_pauses can require a longer pause at
    particular cuts, e.g. a sentence break after a context line, so a cut never lands in the gap between two words."""
    env = envelope(audio)
    if len(env) == 0 or len(weights) < 2:
        return None
    quiet = ~sounding(env)  # cut only inside true silence, never through a soft consonant
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
    min_pauses = min_pauses or [min_pause] * (len(weights) - 1)
    span = last - first
    total = float(sum(weights))
    cuts, prev, acc = [], first, 0.0
    for i, w in enumerate(weights[:-1]):
        acc += w
        expected = first + span * acc / total
        candidates = [(a, b) for a, b in runs if (a + b) / 2 > prev and abs((a + b) / 2 - expected) <= tolerance * span
                      and (b - a + 1) * 0.01 >= min_pauses[i]]
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
    for q in ~sounding(env):
        run = run + 1 if q else 0
        longest = max(longest, run)
    return longest * FRAME / SAMPLE_RATE


class GlitchDetector:
    """Spots clips where the TTS model went wrong: a long silence in the middle of speech, or a clip far too short or
    long for its text compared with how fast that voice usually speaks (dropped or repeated words)."""
    MAX_PAUSE = 1.6

    def __init__(self, verifier=None):
        self.rates = {}  # voice -> (chars, seconds) of clips that passed
        self.verifier = verifier  # optional audiblez.verify.Verifier, which also listens for dropped/invented words

    def problem(self, audio, text, voice):
        seconds = len(audio) / SAMPLE_RATE
        if seconds == 0:
            return 'no audio'
        pause = longest_pause(audio)
        if pause > self.MAX_PAUSE:
            return f'{pause:.1f}s silence inside the clip'
        chars, total = self.rates.get(voice, (0, 0.0))
        if chars >= 300 and len(text) >= 40:
            expected = spoken_length(text) * total / chars
            if not 0.6 <= seconds / expected <= 1.7:
                return f'{seconds:.1f}s of audio for text that usually takes {expected:.1f}s'
        return None

    def accept(self, audio, text, voice):
        chars, total = self.rates.get(voice, (0, 0.0))
        self.rates[voice] = (chars + spoken_length(text), total + len(audio) / SAMPLE_RATE)


def spoken_length(text):
    """Length of the text as spoken, roughly: a digit takes as long to say as a few letters ("28:20")."""
    return len(text) + 3 * sum(c.isdigit() for c in text)


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
