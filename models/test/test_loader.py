import unittest

from parameterized import parameterized

from models.keypaper.loader import Loader


class TestLoader(unittest.TestCase):

    @parameterized.expand([
        ('FooBar', '"\'FooBar\'"'),
        ('Foo Bar', '"\'Foo\' AND \'Bar\'"'),
        ('"Foo Bar"', '\'"Foo Bar"\''),
        ('"Foo" Bar"', '\'"Foo Bar"\''),
        ('Foo-Bar', '"\'Foo-Bar\'"'),
        ('&^Foo-Bar', '"\'Foo-Bar\'"'),
    ])
    def test_valid_source(self, terms, expected):
        self.assertEqual(expected, Loader.preprocess_search_string(terms, 0))

    def test_too_many_words(self):
        self.assertEqual('"\'Foo\'"', Loader.preprocess_search_string('Foo', 1))
        with self.assertRaises(Exception):
            self.assertEqual(Loader.preprocess_search_string('Foo', 2), '')


if __name__ == '__main__':
    unittest.main()
