import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from models.keypaper.utils import split_df_list


class TestUtils(unittest.TestCase):
    def test_split_df_list(self):
        data_for_df = [[2, 'a, b, c'],
                       [1, 'c, a, d'],
                       [4, 'd, c']]

        df_with_list_column = pd.DataFrame(data_for_df, columns=['id', 'list'])

        expected_data = [[2, 'a'], [2, 'b'], [2, 'c'],
                         [1, 'c'], [1, 'a'], [1, 'd'],
                         [4, 'd'], [4, 'c']]
        expected_df = pd.DataFrame(expected_data, columns=['id', 'list'])
        actual_df = split_df_list(df_with_list_column, target_column='list', separator=', ')
        assert_frame_equal(expected_df, actual_df, "Splitting list into several rows works incorrectly")
