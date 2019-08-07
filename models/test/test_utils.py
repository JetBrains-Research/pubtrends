import unittest

from models.keypaper.utils import tokenize


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
