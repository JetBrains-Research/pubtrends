import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal
from parameterized import parameterized
from time import sleep

from models.celery.tasks_cache import lru_ttl_cache_with_callback
from models.keypaper.utils import tokenize, cut_authors_list, split_df_list, crc32


removed = []


def remove_callback(k):
    removed.append(k)


@lru_ttl_cache_with_callback(maxsize=2, timeout=1, remove_callback=remove_callback)
def foo(x):
    return x + 1


class TestUtils(unittest.TestCase):

    def test_ttl_cache_with_callback(self):
        self.assertEqual(0, foo.size())
        foo(1)
        foo(2)
        foo(1)
        foo(2)
        foo(1)
        foo(2)
        self.assertEqual(2, foo.size())
        self.assertEqual([1, 2], foo.keys())
        sleep(2)
        # Access required to revoke items
        self.assertEqual(2, foo.size())
        self.assertEqual([], removed)
        foo(3)
        self.assertEqual(1, foo.size())
        foo(4)
        foo(5)
        self.assertEqual(2, foo.size())
        self.assertEqual([4, 5], foo.keys())
        self.assertEqual([1, 2, 3], removed)
