import unittest

import numpy as np
from bokeh.models import ColumnDataSource

from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.plot.plotter import Plotter
from pysrc.test.plot.mock_analyzer import MockAnalyzer


class TestPlotPreprocessor(unittest.TestCase):

    def setUp(self):
        self.analyzer = MockAnalyzer()
        self.plotter = Plotter(self.analyzer)

    def test_component_size_summary(self):
        components, data = PlotPreprocessor.component_size_summary_data(
            self.analyzer.df, self.analyzer.components, self.analyzer.min_year, self.analyzer.max_year
        )

        expected_components = ['1', '2', '3']
        expected_components_data = {
            # from 2005 to 2019
            # 0 : {2008: 1, 2013: 1, 2017: 3, 2018: 1}
            '1': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 3, 1, 0],
            # 1 : {2005: 1, 2009: 1, 2011: 1, 2016: 1}
            '2': [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            # 2 : {2019: 2}
            '3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        self.assertEqual(components, expected_components, 'Wrong list of components')
        self.assertEqual(min(data['years']), self.analyzer.min_year, 'Wrong min year')
        self.assertEqual(max(data['years']), self.analyzer.max_year, 'Wrong max year')

        for c in expected_components_data.keys():
            self.assertEqual(data[c], expected_components_data[c], f'Wrong component size list for component {c}')

    def test_paper_statistics_data(self):
        ds = ColumnDataSource(PlotPreprocessor.papers_statistics_data(self.analyzer.df))

        expected_years = list(range(1970, 2020))
        expected_counts = [1] + [0] * 34 + [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 6, 2, 2]

        self.assertEqual(list(ds.data['year']), expected_years, 'Wrong list of years')
        self.assertEqual(list(ds.data['counts']), expected_counts, 'Wrong list of paper counts')

    def test_article_view_data_source_split(self):
        width = 760
        lbefore = len(set(zip(self.analyzer.df['year'], self.analyzer.df['total'])))
        ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
            self.analyzer.df, self.analyzer.min_year, self.analyzer.max_year, True, width=width
        ))
        lafter = len(set(zip(ds.data['year'], ds.data['total'])))
        self.assertGreaterEqual(lafter, lbefore)

        self.assertFalse(np.any(ds.data['year'] == np.nan), 'NaN values in `year` column')
        self.assertFalse(np.any(ds.data['size'] == np.nan), 'NaN values in `size` column')

        max_size = ds.data['size'].max()
        max_width = max_size * (self.analyzer.max_year - self.analyzer.min_year + 1)
        self.assertLessEqual(max_width, width, 'Horizontal overlap')

    def test_article_view_data_source(self):
        width = 760
        lbefore = len(set(zip(self.analyzer.df['year'], self.analyzer.df['total'])))
        ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
            self.analyzer.df, self.analyzer.min_year, self.analyzer.max_year, False, width=width
        ))

        lafter = len(set(zip(ds.data['year'], ds.data['total'])))
        self.assertGreaterEqual(lafter, lbefore)

        self.assertFalse(np.any(ds.data['year'] == np.nan), 'NaN values in `year` column')
        self.assertFalse(np.any(ds.data['size'] == np.nan), 'NaN values in `size` column')

        max_size = ds.data['size'].max()
        max_width = max_size * (self.analyzer.max_year - self.analyzer.min_year + 1)
        self.assertLessEqual(max_width, width, 'Horizontal overlap')

    def test_topic_evolution_data(self):
        edges, nodes_data = PlotPreprocessor.topic_evolution_data(
            self.analyzer.evolution_df, self.analyzer.evolution_kwds, self.analyzer.n_steps
        )

        expected_edges = [('2021 1', '2045 1', 3), ('2021 1', '2045 2', 2), ('2021 NPY', '2045 1', 1),
                          ('2021 NPY', '2045 2', 4)]
        expected_nodes_data = [('2045 1', '2045 1 shiftwork,estrogen,pattern'),
                               ('2045 2', '2045 2 study,analysis,association'), ('2021 NPY', 'NPY'),
                               ('2021 1', '2021 1 rassf1a,tsg,datasets')]
        self.assertListEqual(nodes_data, expected_nodes_data, 'Wrong nodes data')
        self.assertCountEqual(edges, expected_edges, 'Wrong Sankey diagram edges')
        self.assertListEqual([el[0] for el in nodes_data], [el[0] for el in expected_nodes_data],
                             'Wrong node order')

    def test_topic_evolution_keywords(self):
        kwds_data = PlotPreprocessor.topic_evolution_keywords_data(
            self.analyzer.evolution_kwds
        )

        expected_keywords_data = [(2021,
                                   1,
                                   'rassf1a, tsg, datasets, biopsy, apc, benign, mutation, measure, history'),
                                  (2045, 1, 'shiftwork, estrogen, pattern, disturbance, cell, per2, disruption'),
                                  (2045, 2, 'study, analysis, association')]
        self.assertEqual(expected_keywords_data, kwds_data)


if __name__ == '__main__':
    unittest.main()
