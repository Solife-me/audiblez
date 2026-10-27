import unittest
from types import SimpleNamespace

from audiblez.article import rule_based, strip_urls, fill_gaps, extend_edges, remove_byline, clean_article
from audiblez.pdf import PdfChapter, Paragraph

BODY = ('In the next week, two sets of elections will be held on two continents, the outcomes of which could have '
        'profound consequences for the world, and both campaigns are watched closely abroad.')


class FakeLLM:
    model = 'fake'

    def __init__(self, answers):
        self.answers = list(answers)

    def chat_json(self, system, user):
        return self.answers.pop(0)


class RulesTest(unittest.TestCase):
    def test_boilerplate_is_dropped(self):
        paragraphs = ['Subscribe Sign In', 'Advertisement', 'Top image credit: Getty.', 'More from InDepth.',
                      'Follow BBC on:', 'Share on Facebook', '© 2026 BBC. All rights reserved.', '5 min read',
                      'By clicking “Accept and Continue,” you confirm you agree to the Terms of Use and the collection '
                      'and use of your information.', BODY]
        self.assertEqual(rule_based(paragraphs), [BODY])

    def test_article_text_mentioning_cookies_is_kept(self):
        recipe = 'Bake the cookies for twelve minutes, until the edges are golden and the centres still soft.'
        self.assertEqual(rule_based([recipe]), [recipe])

    def test_urls_and_isbns_are_removed(self):
        self.assertEqual(strip_urls('See https://www.bbc.com/news/articles/crlykey9jywqo for more.'), 'See for more.')
        self.assertEqual(strip_urls('All of Grace, ISBN 978-1-56186-211-5, is a classic.'), 'All of Grace, , is a classic.')


class SelectionTest(unittest.TestCase):
    def test_gaps_are_filled_with_prose_but_not_captions_or_notices(self):
        candidates = [BODY, 'A caption.', BODY + ' A quotation.', 'We use cookies and similar technologies ' * 6, BODY]
        self.assertEqual(fill_gaps(candidates, [0, 4]), [0, 2, 4])

    def test_closing_paragraph_is_recovered_at_the_edge(self):
        candidates = ['Junk menu line that is long enough to be considered by the edge check, really.', BODY, BODY,
                      'The choices voters make in coming days may shape where the world goes next, for years.',
                      'BBC InDepth is the home on the website and app for the best analysis and deep reporting.']
        llm = FakeLLM([{'belongs': True}, {'belongs': False}, {'belongs': False}])
        self.assertEqual(extend_edges(llm, candidates, [1, 2]), [1, 2, 3])

    def test_byline_glued_to_the_first_paragraph_is_removed(self):
        self.assertEqual(remove_byline('James Landale In the next week, two sets...', ['James Landale']),
                         'In the next week, two sets...')

    def test_clean_article_with_llm(self):
        paragraphs = ['Subscribe Sign In', 'Israelis and Americans are both voting', 'James Landale ' + BODY, BODY,
                      'More from InDepth.', 'Why Israel\'s election is one of the most important in its history.']
        book = SimpleNamespace(title='Page title', creator='', cover_image=b'',
                               chapters=[PdfChapter('x', [Paragraph(p, 0) for p in paragraphs])])
        llm = FakeLLM([{'title': 'Israelis and Americans are both voting', 'authors': ['James Landale'],
                        'date': '26 October 2026', 'body': [[1, 2]]},
                       {'belongs': False}, {'belongs': False}])
        out = clean_article(book, llm, log=lambda *a: None)
        texts = [p.text for p in out.chapters[0].paragraphs]
        self.assertEqual(texts[0], 'Israelis and Americans are both voting. By James Landale. 26 October 2026.')
        self.assertEqual(texts[1:], [BODY, BODY])
        self.assertTrue(out.intro_included)

    def test_without_llm_only_rules_apply(self):
        book = SimpleNamespace(title='T', creator='A', cover_image=b'',
                               chapters=[PdfChapter('x', [Paragraph(p, 0) for p in ['Advertisement', BODY]])])
        self.assertEqual([p.text for p in clean_article(book, None, log=lambda *a: None).chapters[0].paragraphs],
                         ['T. By A.', BODY])


if __name__ == '__main__':
    unittest.main()
