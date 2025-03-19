import unittest
from parameterized import parameterized

from pysrc.papers.utils import cut_authors_list, crc32, preprocess_doi, rgb2hex


class TestUtils(unittest.TestCase):

    def test_cut_authors_list_limit_size(self):
        limit_size_list = "first, second, third"
        actual = cut_authors_list(limit_size_list, limit=len(limit_size_list))
        self.assertEqual(actual, limit_size_list)

    def test_cut_authors_list_less(self):
        long_list = "first, second, third"
        actual = cut_authors_list(long_list, 2)
        expected = "first,...,third"
        self.assertEqual(actual, expected)

    def test_cut_authors_list_greater(self):
        short_list = "first, second, third"
        actual = cut_authors_list(short_list, 4)
        self.assertEqual(actual, short_list)

    @parameterized.expand([
        ('cc77a65ff80a9d060e48461603bcf06bb0ef9294', -189727251, 'negative'),
        ('6d8484217c9fa02419536c9118435715d3a8e705', 1979136599, 'positive')
    ])
    def test_crc32(self, ssid, crc32id, case):
        self.assertEqual(crc32(ssid), crc32id, f"Hashed id is wrong ({case} case)")

    @parameterized.expand([
        ('dx.doi.org prefix', 'http://dx.doi.org/10.1037/a0028240', '10.1037/a0028240'),
        ('doi.org prefix', 'http://doi.org/10.3352/jeehp.2013.10.3', '10.3352/jeehp.2013.10.3'),
        ('no changes', '10.1037/a0028240', '10.1037/a0028240')
    ])
    def test_preprocess_doi(self, case, doi, expected):
        self.assertEqual(preprocess_doi(doi), expected, case)

    @parameterized.expand([
        ([145, 200, 47], '#91c82f'),
        ([143, 254, 9], '#8ffe09'),
        ('red', '#ff0000'),
        ('blue', '#0000ff')
    ])
    def test_color2hex(self, color, expected):
        self.assertEqual(rgb2hex(color), expected)
