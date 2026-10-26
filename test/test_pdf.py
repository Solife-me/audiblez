import os
import unittest
from pathlib import Path

from audiblez import pdf
from audiblez.pdf import Line, PdfChapter, Paragraph, build_paragraphs, cap_length, join_lines


def line(text, page=0, x0=108, top=100, size=11, x1=504):
    return Line(text, page, x0, x1, top, top + size, size)


class ParagraphTest(unittest.TestCase):
    def test_indent_starts_paragraph_and_hyphenation_is_undone(self):
        lines = [line('First paragraph that runs the full width and is hyphen-', top=100),
                 line('ated across lines.', top=115, x1=200),
                 line('Second paragraph starts indented.', x0=126, top=130)]
        paras = build_paragraphs(lines, 11)
        self.assertEqual([p.text for p in paras], [
            'First paragraph that runs the full width and is hyphenated across lines.',
            'Second paragraph starts indented.'])

    def test_paragraph_continues_across_page_break(self):
        lines = [line('A sentence that does not end on this', page=0, top=700),
                 line('page but continues on the next one.', page=1, top=80, x1=300)]
        self.assertEqual(len(build_paragraphs(lines, 11)), 1)

    def test_vertical_gap_starts_paragraph(self):
        lines = [line('Block style paragraph one.', top=100, x1=250), line('Block style paragraph two.', top=140)]
        self.assertEqual(len(build_paragraphs(lines, 11)), 2)

    def test_large_font_is_heading(self):
        paras = build_paragraphs([line('Chapter One', size=18, top=50), line('Body text.', top=100)], 11)
        self.assertTrue(paras[0].heading)
        self.assertEqual(paras[0].text, 'Chapter One.')

    def test_join_lines_keeps_dashes_tight(self):
        self.assertEqual(join_lines(['God is unchanging—', 'in his love']), 'God is unchanging—in his love')

    def test_cap_length_splits_between_paragraphs(self):
        chapter = PdfChapter('Long', [Paragraph('x' * 300, 0) for _ in range(10)])
        parts = cap_length(chapter, 1000)
        self.assertEqual([len(p.paragraphs) for p in parts], [3, 3, 3, 1])
        self.assertEqual(parts[0].title, 'Long (part 1)')


class SpurgeonPdfTest(unittest.TestCase):
    def test_outline_chapters(self):
        name = 'spurgeon.pdf'
        if not Path(name).exists():
            os.system(f'wget https://ccel.org/ccel/s/spurgeon/sermons01/cache/sermons01.pdf -O {name}')
        book = pdf.load_pdf(name)
        self.assertEqual(book.creator, 'Charles Spurgeon')
        self.assertEqual(book.good_chapters[0].title, 'Sermon 1. The Immutability of God')
        text = book.good_chapters[0].extracted_text
        self.assertNotIn('Sermon 1. The Immutability of God\n', text[200:])  # running header removed
        self.assertIn('proper study of God’s elect is God', text)


if __name__ == '__main__':
    unittest.main()
