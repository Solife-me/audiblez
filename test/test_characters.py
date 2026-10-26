import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from audiblez.characters import (Registry, Character, split_dialogue, detect_quote_style, assign_voices,
                                 prepare_cast, read_cast_sheet, attribute_chapter, paragraph_spans)
from audiblez.tts import TTSEngine, VoiceInfo


class SplitDialogueTest(unittest.TestCase):
    def quotes(self, text, style):
        return [s.text for s in split_dialogue(text, style) if s.is_quote]

    def test_curly_quotes_with_dialogue_tag(self):
        text = '“My dear Mr. Bennet,” said his lady to him one day, “have you heard?”'
        self.assertEqual(self.quotes(text, 'double'), ['“My dear Mr. Bennet,”', '“have you heard?”'])

    def test_quote_continuing_into_next_paragraph_stays_open(self):
        spans = split_dialogue('“This is a long speech that goes on', 'double')
        self.assertEqual([s.is_quote for s in spans], [True])

    def test_british_single_quotes_ignore_apostrophes(self):
        text = '‘I don’t know,’ said Harry’s friend. ‘It’s the boys’ turn.’'
        self.assertEqual(self.quotes(text, 'single'), ['‘I don’t know,’', '‘It’s the boys’ turn.’'])
        self.assertEqual(detect_quote_style(text), 'single')

    def test_straight_quotes(self):
        self.assertEqual(self.quotes('"Yes," he said. "No."', 'straight'), ['"Yes,"', '"No."'])


class RegistryTest(unittest.TestCase):
    def test_titles_keep_spouses_apart(self):
        r = Registry()
        r.add('Mr. Bennet', 'male')
        r.add('Mrs. Bennet', 'female')
        self.assertEqual(r.resolve('Mrs. Bennet'), 'Mrs. Bennet')
        self.assertEqual(r.resolve('Mr. Bennet'), 'Mr. Bennet')
        self.assertEqual(len(r.characters), 2)

    def test_partial_names_merge(self):
        r = Registry()
        r.add('Darcy', 'male')
        self.assertEqual(r.resolve('Mr. Darcy'), 'Mr. Darcy')  # renamed to the fuller name
        self.assertEqual(r.resolve('Darcy'), 'Mr. Darcy')
        r.add('Elizabeth Bennet', 'female', aliases=['Lizzy'])
        self.assertEqual(r.resolve('Lizzy'), 'Elizabeth Bennet')
        self.assertEqual(r.resolve('Elizabeth'), 'Elizabeth Bennet')

    def test_alias_merges_existing_character(self):
        r = Registry()
        r.add('Lizzy', 'female')
        r.add('Elizabeth', 'female', aliases=['Lizzy'])
        self.assertEqual(list(r.characters), ['Elizabeth'])
        self.assertEqual(r.resolve('Lizzy'), 'Elizabeth')

    def test_gender_from_title(self):
        r = Registry()
        self.assertEqual(r.characters[r.add('Lady Catherine')].gender, 'female')


class FakeLLM:
    model = 'fake'

    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def chat_json(self, system, user):
        self.calls += 1
        return self.answers.pop(0)


class FakeEngine(TTSEngine):
    name = 'fake'

    def list_voices(self):
        return {v.name: v for v in [VoiceInfo('narr', 'male', 'british'), VoiceInfo('m1', 'male', 'british'),
                                    VoiceInfo('m2', 'male', 'american'), VoiceInfo('f1', 'female', 'british'),
                                    VoiceInfo('ogre', 'male', 'british', style='character')]}


class CastingTest(unittest.TestCase):
    def test_assign_voices_prefers_unused_voice_of_same_gender_and_accent(self):
        chars = [Character('A', 'male', lines=10), Character('B', 'male', lines=5), Character('C', 'female', lines=3),
                 Character('D', 'unknown', lines=1), Character('E', 'male', lines=1)]
        assign_voices(chars, FakeEngine().list_voices(), 'narr', 'fake', accent='british')
        self.assertEqual([c.voice for c in chars], ['m1', 'm2', 'f1', 'narr', 'm1'])

    def test_prepare_cast_writes_sheet_then_reuses_cache(self):
        text = '“Hello,” said Anne.\nTom nodded. “Hi.”\nThe end.'
        chapter = SimpleNamespace(extracted_text=text, get_name=lambda: 'ch1')
        book = SimpleNamespace(title='T', creator='C')
        llm = FakeLLM([{'quotes': {'0': 'Anne', '1': 'Tom'},
                        'characters': [{'name': 'Anne', 'gender': 'female'}, {'name': 'Tom', 'gender': 'male'}]}])
        with tempfile.TemporaryDirectory() as out:
            cast, is_new = prepare_cast([chapter], book, FakeEngine(), 'narr', llm, out, 'book', log=lambda *a: None,
                                        llm_casting=False)
            self.assertTrue(is_new)
            narrator, characters = read_cast_sheet(Path(out) / 'book.cast.toml')
            self.assertEqual(narrator, 'narr')
            self.assertEqual({c.name: c.voice for c in characters}, {'Anne': 'f1', 'Tom': 'm1'})
            chunks = cast.chunks(text, 1000)
            self.assertEqual([(c.text, c.voice) for c in chunks], [
                ('“Hello,”', 'f1'), ('said Anne.', 'narr'), ('Tom nodded.', 'narr'), ('“Hi.”', 'm1'), ('The end.', 'narr')])
            # Second run: cached attributions, no LLM calls, and the edited cast sheet is honoured
            sheet = Path(out) / 'book.cast.toml'
            sheet.write_text(sheet.read_text().replace('voice = "m1"', 'voice = "m2"'))
            cast, is_new = prepare_cast([chapter], book, FakeEngine(), 'narr', llm, out, 'book', log=lambda *a: None,
                                        llm_casting=False)
            self.assertFalse(is_new)
            self.assertEqual(llm.calls, 1)
            self.assertEqual(cast.voice_for('Tom'), 'm2')

    def test_llm_casting_is_validated(self):
        from audiblez.characters import llm_cast
        chars = [Character('Mrs. Bennet', 'female', lines=30, description='excitable mother'),
                 Character('Mr. Bennet', 'male', lines=25), Character('Mr. Collins', 'male', lines=12),
                 Character('Lady Lucas', 'female', lines=1)]
        llm = FakeLLM([{'cast': {'Mrs. Bennet': {'voice': 'f1', 'why': 'lively'},
                                 'Mr. Bennet': {'voice': 'f1', 'why': 'wrong gender'},     # rejected: female voice
                                 'Mr. Collins': {'voice': 'm1'},
                                 'Lady Lucas': {'voice': 'f1'}}}])                         # minor: may share
        n = llm_cast(llm, chars, FakeEngine().list_voices(), 'narr', 'fake', log=lambda *a: None)
        self.assertEqual(n, 3)
        self.assertEqual([c.voice for c in chars], ['f1', '', 'm1', 'f1'])
        self.assertEqual(chars[0].why, 'lively')
        assign_voices(chars, FakeEngine().list_voices(), 'narr', 'fake')  # the fallback fills the rest
        self.assertEqual(chars[1].voice, 'm2')

    def test_llm_casting_keeps_main_characters_apart(self):
        from audiblez.characters import llm_cast
        chars = [Character('A', 'male', lines=30), Character('B', 'male', lines=20)]
        llm = FakeLLM([{'cast': {'A': {'voice': 'm1'}, 'B': {'voice': 'm1'}}}])
        llm_cast(llm, chars, FakeEngine().list_voices(), 'narr', 'fake', log=lambda *a: None)
        self.assertEqual([c.voice for c in chars], ['m1', ''])

    def test_bad_llm_answer_falls_back_to_narrator(self):
        spans = paragraph_spans('“Hi,” she said.', 'double')
        speakers = attribute_chapter(FakeLLM([{'nonsense': True}]), spans, Registry(), log=lambda *a: None)
        self.assertEqual(speakers, ['unknown'])


if __name__ == '__main__':
    unittest.main()
