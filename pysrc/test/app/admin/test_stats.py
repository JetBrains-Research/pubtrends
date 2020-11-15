import unittest

from parameterized import parameterized

from pysrc.app.admin.stats import duration_seconds


class TestStats(unittest.TestCase):
    @parameterized.expand([
        (0, '00:00:00'),
        (31, '00:00:31'),
        (62, '00:01:02'),
        (363, '00:06:03'),
        (3604, '01:00:04'),
        (10000, '02:46:40')
    ])
    def test_duration(self, seconds, result):
        self.assertEqual(result, duration_seconds(seconds))

