import unittest

from audiblez.text import chunk_text, split_long_sentence, split_paragraphs


class ChunkTextTest(unittest.TestCase):
    def test_chunks_never_cross_paragraphs(self):
        text = 'One. Two.\nThree, four. Five.\n\nSix.'
        chunks = chunk_text(text, 'v', 1000)
        self.assertEqual([c.text for c in chunks], ['One. Two.', 'Three, four. Five.', 'Six.'])
        self.assertTrue(all(c.paragraph_end for c in chunks))

    def test_long_paragraph_splits_at_sentences(self):
        sentence = 'This sentence is about fifty characters long, ok. '
        chunks = chunk_text(sentence * 10, 'v', 120)
        self.assertTrue(all(len(c.text) <= 120 for c in chunks))
        self.assertTrue(all(c.text.endswith('ok.') for c in chunks))
        self.assertEqual([c.paragraph_end for c in chunks], [False] * (len(chunks) - 1) + [True])

    def test_long_sentence_prefers_clause_breaks(self):
        text = 'alpha beta gamma, ' * 30
        parts = split_long_sentence(text.strip(), 100)
        self.assertTrue(all(len(p) <= 100 for p in parts))
        self.assertTrue(all(p.endswith(',') for p in parts[:-1]))
        self.assertEqual(' '.join(parts), text.strip())

    def test_split_paragraphs_drops_blank_lines(self):
        self.assertEqual(split_paragraphs('\n a \n\n\n b\n'), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
