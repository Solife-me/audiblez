import unittest

import numpy as np

from audiblez.audio import trim, split_at_pauses, Leveler, GlitchDetector
from audiblez.characters import Cast, Registry, Character, text_key
from audiblez.tts import SAMPLE_RATE


def tone(seconds, amplitude=0.3, freq=150):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def gap(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class AudioTest(unittest.TestCase):
    def test_trim_removes_padding(self):
        clip = np.concatenate([gap(0.3), tone(1.0), gap(0.7)])
        self.assertAlmostEqual(len(trim(clip)) / SAMPLE_RATE, 1.06, delta=0.03)

    def test_split_at_the_pause_matching_the_weights(self):
        # two pauses; the weights say the cut belongs after the first second, not after the second
        audio = np.concatenate([tone(1.0), gap(0.2), tone(0.5), gap(0.25), tone(0.5)])
        pieces = split_at_pauses(audio, [1.0, 1.0])
        self.assertIsNotNone(pieces)
        self.assertAlmostEqual(len(trim(pieces[0])) / SAMPLE_RATE, 1.03, delta=0.02)  # 1s tone + trailing margin

    def test_split_gives_up_without_a_clear_pause(self):
        self.assertIsNone(split_at_pauses(tone(2.0), [1.0, 1.0]))
        # a pause exists, but far from where the weights put the cut
        self.assertIsNone(split_at_pauses(np.concatenate([tone(0.3), gap(0.2), tone(1.7)]), [1.0, 1.0]))

    def test_leveler_matches_loudness(self):
        lev = Leveler()
        quiet, loud = lev(tone(1.0, 0.05), 'a'), lev(tone(1.0, 0.5), 'b')
        self.assertAlmostEqual(np.sqrt(np.mean(quiet ** 2)), np.sqrt(np.mean(loud ** 2)), delta=0.005)


class GlitchTest(unittest.TestCase):
    def test_long_internal_silence_is_a_glitch(self):
        g = GlitchDetector()
        self.assertIsNone(g.problem(np.concatenate([tone(1), gap(0.8), tone(1)]), 'x' * 30, 'v'))
        self.assertIn('silence', g.problem(np.concatenate([tone(1), gap(4.0), tone(1)]), 'x' * 30, 'v'))

    def test_clip_too_short_for_its_text_is_a_glitch(self):
        g = GlitchDetector()
        for _ in range(4):
            g.accept(tone(5.0), 'x' * 80, 'v')  # this voice reads 16 chars/s
        self.assertIsNone(g.problem(tone(5.0), 'x' * 80, 'v'))
        self.assertIn('usually takes', g.problem(tone(1.5), 'x' * 80, 'v'))  # words were dropped


class VerifierTest(unittest.TestCase):
    def verifier(self, heard):
        from audiblez.verify import Verifier
        v = Verifier()
        v.transcribe = lambda audio: heard
        return v

    def test_recognizer_slips_are_ignored(self):
        v = self.verifier('Tomorrow, Fortnite, said Mr. Bennett to his wife.')
        self.assertIsNone(v.problem(tone(1), 'To-morrow fortnight, said Mister Bennet to his wife.'))

    def test_dropped_words_are_caught(self):
        v = self.verifier('for after all, Mr. Self.')
        wrong, detail = v.problem(tone(1), 'for after all, Mister Self is the worst enemy a Christian has.')
        self.assertEqual(wrong, 7)  # "is the worst enemy a christian has"
        self.assertIn('dropped', detail)

    def test_invented_words_are_caught(self):
        v = self.verifier('from an upper window, that he had to be at the end of the city, that he wore a blue coat')
        self.assertIn('added', v.problem(tone(1), 'from an upper window, that he wore a blue coat')[1])


class SplitRetryTest(unittest.TestCase):
    def test_clip_that_keeps_looping_is_synthesized_in_shorter_pieces(self):
        from audiblez.core import synthesize_clip
        from audiblez.tts import TTSEngine

        class LoopingEngine(TTSEngine):  # like phonon on one long sentence: always repeats itself
            name = 'loop'
            calls = []

            def synthesize(self, text, voice, speed=1.0):
                self.calls.append(text)
                return tone(len(text) / 16)

        class StrictVerifier:
            def applies(self, engine, voice): return True
            def problem(self, audio, text): return (5, 'added "he loves them"') if len(text) > 300 else None

        text = ' '.join(f'Sentence number {i} is here, and it goes on for a while.' for i in range(12))
        engine = LoopingEngine()
        audio = synthesize_clip(engine, text, 'v', 1.0, GlitchDetector(StrictVerifier()))
        self.assertEqual(engine.calls[:2], [text, text])  # two tries as it is
        self.assertTrue(all(len(t) <= 300 for t in engine.calls[2:]))  # then shorter pieces
        self.assertEqual(' '.join(engine.calls[2:]), text)  # covering all the text, once
        self.assertGreater(len(audio), len(tone(len(text) / 16)))  # the pieces plus the pauses between them


class DialogueChunkTest(unittest.TestCase):
    def cast(self, text, speakers):
        chars = [Character('Mrs. Bennet', 'female', voice='f1'), Character('Mr. Bennet', 'male', voice='m1')]
        return Cast('', '', 'narr', Registry(chars), 'double', {text_key(text): speakers})

    def test_split_quote_is_one_utterance(self):
        text = '“My dear Mr. Bennet,” said his lady to him one day, “have you heard?”'
        chunks = self.cast(text, ['Mrs. Bennet', 'Mrs. Bennet']).chunks(text, 1000)
        self.assertEqual([c.voice for c in chunks], ['f1', 'narr', 'f1'])
        self.assertEqual(chunks[0].joint, chunks[2].joint)
        self.assertEqual((chunks[0].joint_index, chunks[2].joint_index), (0, 1))
        self.assertIsNone(chunks[1].joint)
        self.assertLess(chunks[0].pause(), 0.2)  # “My dear Mr. Bennet,” flows straight into the tag

    def test_dialogue_tag_is_primed_with_the_end_of_the_quote(self):
        text = '“My dear Mr. Bennet,” said his lady to him one day, “have you heard?”'
        chunks = self.cast(text, ['Mrs. Bennet', 'Mrs. Bennet']).chunks(text, 1000, prime_replies=True)
        tag = chunks[1]
        self.assertEqual(tag.voice, 'narr')
        self.assertEqual(tag.joint, ('My dear Mister Bennet,', 'said his lady to him one day,'))
        self.assertTrue(tag.joint_context)

    def test_prime_replies_adds_the_line_being_answered(self):
        text = '“Is he married or single?”\n“Oh, single, my dear!”'
        chunks = self.cast(text, ['Mr. Bennet', 'Mrs. Bennet']).chunks(text, 1000, prime_replies=True)
        reply = chunks[1]
        self.assertEqual(reply.joint, ('Is he married or single?', '“Oh, single, my dear!”'))
        self.assertTrue(reply.joint_context)
        self.assertEqual(reply.joint_index, 1)
        self.assertLess(chunks[0].pause(), 0.6)  # a turn in a conversation, not a narration paragraph break


if __name__ == '__main__':
    unittest.main()
