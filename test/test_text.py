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


class RomanNumeralTest(unittest.TestCase):
    def test_numerals_are_read_as_numbers_where_context_is_clear(self):
        from audiblez.text import normalize_for_speech as n
        cases = {
            'CHAPTER XII.': 'CHAPTER 12.', 'Book IV': 'Book 4', 'Act V, Scene ii.': 'Act 5, Scene 2.',
            'II. Now secondly, let us': '2. Now secondly, let us', 'III, Our third point': '3, Our third point',
            'heaven.” II. The second part': 'heaven.” 2. The second part',
            'When Charles II ordered': 'When Charles the Second ordered', 'Elizabeth I of England': 'Elizabeth the First of England',
            'World War II ended': 'World War 2 ended', 'VIII., IX., X.': '8., 9., 10.',
            'In Isaiah, chap. lxiii, v. 10, it is said': 'In Isaiah, chapter 63, verse 10, it is said',
            'As in 1 Cor. xii.12, and': 'As in First Corinthians 12:12, and', 'Mal. iii. 6': 'Malachi 3:6',
            'Rev. Smith preached on Rev. 3:20.': 'Reverend Smith preached on Revelation 3:20.',
            'the young ladies.{vii} She said': 'the young ladies. She said',
        }
        for text, expected in cases.items():
            self.assertEqual(n(text), expected, text)

    def test_pronoun_and_words_made_of_numeral_letters_are_left_alone(self):
        from audiblez.text import normalize_for_speech as n
        for text in ['I am sure I will. Then I said, I know.', 'Grand Rapids, MI: Christian Classics', 'a free CD, please',
                     'MIX the DIM light', 'an ill-omen', 'Elizabeth I went home', 'In this discourse I shall endeavor',
                     'for the most part I can cry', 'one book I love to read', 'A VIVID CIVIL war']:
            self.assertEqual(n(text), text)

    def test_numerals_can_be_left_as_written(self):
        from audiblez.text import normalize_for_speech as n
        self.assertEqual(n('CHAPTER XII. Mr. Smith', roman_numerals=False), 'CHAPTER XII. Mister Smith')


class FootnoteTest(unittest.TestCase):
    HTML = """<body><p>The people objected<a href="#fn1" class="noteref">1</a> at once.<sup>2</sup> On the 19<sup>th</sup>.</p>
    <p>See [<a href="#note3">3</a>] and <a epub:type="noteref" href="#n4">*</a> here.</p>
    <p class="foot">1 A Gutenberg footnote.</p><aside epub:type="footnote"><p>An EPUB 3 footnote.</p></aside>
    <section role="doc-endnotes"><p>Endnotes.</p></section><div class="footnote"><p>Another note.</p></div></body>"""

    def text(self, skip):
        from bs4 import BeautifulSoup
        from audiblez.core import strip_footnotes
        soup = BeautifulSoup(self.HTML, 'lxml')
        if skip:
            strip_footnotes(soup)
        import re
        return [re.sub(r'\s*\[\s*\]', '', re.sub(r'\s+', ' ', p.text)).strip() for p in soup.find_all('p')]

    def test_bodies_and_markers_are_removed(self):
        self.assertEqual(self.text(skip=True), ['The people objected at once. On the 19th.', 'See and here.'])

    def test_kept_when_disabled(self):
        self.assertEqual(len(self.text(skip=False)), 6)


if __name__ == '__main__':
    unittest.main()
