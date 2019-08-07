import unittest

from models.keypaper.utils import tokenize, cut_authors_list


class TestUtils(unittest.TestCase):
    def test_tokenizer(self):
        text = """Very interesting article about elephants and donkeys.
        There are two types of elephants - Indian and African.
        Both of them are really beautiful, but in my opinion Indian are even cuter"""
        # nouns and adjectives from text excluding comparative and superlative forms
        expected = ['interesting', 'article', 'elephant', 'donkey',
                    'type', 'elephant', 'indian', 'african', 'beautiful',
                    'opinion', 'indian']
        actual = tokenize(text)
        self.assertSequenceEqual(actual, expected)

    def test_cut_authors_list_limit_size(self):
        limit_size_list = "first, second, third"
        actual = cut_authors_list(limit_size_list, limit=len(limit_size_list))
        self.assertEquals(actual, limit_size_list)

    def test_cut_authors_list_less(self):
        long_list = "first, second, third"
        actual = cut_authors_list(long_list, 2)
        expected = "first,...,third"
        self.assertEquals(actual, expected)

    def test_cut_authors_list_greater(self):
        short_list = "first, second, third"
        actual = cut_authors_list(short_list, 4)
        self.assertEquals(actual, short_list)