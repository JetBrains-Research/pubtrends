import unittest

import numpy as np
from bokeh.models import ColumnDataSource

from models.keypaper.plot_preprocessor import PlotPreprocessor
from models.keypaper.plotter import Plotter
from models.test.mock_analyzer import MockAnalyzer


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

    def test_component_ratio(self):
        comps, ratios = PlotPreprocessor.component_ratio_data(self.analyzer.df)
        colors = [self.plotter.comp_palette[int(c) - 1] for c in comps]
        source = ColumnDataSource(data=dict(comps=comps, ratios=ratios, colors=colors))

        expected_comps = ['3', '2', '1']
        expected_ratios = [9.090909, 36.363636, 54.545454]
        expected_colors = list(reversed(self.plotter.comp_palette))

        self.assertEqual(comps, expected_comps, 'Wrong list of components')
        for ratio, expected in zip(source.data['ratios'], expected_ratios):
            self.assertAlmostEqual(ratio, expected, places=3, msg='Wrong component ratio')
        self.assertEqual(source.data['colors'], expected_colors, 'Wrong list of component colors')

    def test_paper_statistics_data(self):
        ds = ColumnDataSource(PlotPreprocessor.papers_statistics_data(self.analyzer.df))

        expected_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        expected_counts = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 6, 2, 2]

        self.assertEqual(list(ds.data['year']), expected_years, 'Wrong list of years')
        self.assertEqual(list(ds.data['counts']), expected_counts, 'Wrong list of paper counts')

    def test_article_view_data_sourceSplit(self):
        width = 760
        lbefore = len(set(zip(self.analyzer.df['year'], self.analyzer.df['total'])))
        ds = ColumnDataSource(PlotPreprocessor.article_view_data_source(
            self.analyzer.df, self.analyzer.min_year, self.analyzer.max_year, True, width=width
        ))
        lafter = len(set(zip(ds.data['year'], ds.data['total'])))
        self.assertGreaterEqual(lafter, lbefore)

        self.assertFalse(np.any(ds.data['year'] == np.nan), 'NaN values in `year` column')
        self.assertFalse(np.any(ds.data['size'] == np.nan), 'NaN values in `size` column')

        max_size = np.max(ds.data['size'])
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

        max_size = np.max(ds.data['size'])
        max_width = max_size * (self.analyzer.max_year - self.analyzer.min_year + 1)
        self.assertLessEqual(max_width, width, 'Horizontal overlap')

    def test_heatmap_clusters(self):
        cluster_edges, clusters = PlotPreprocessor.heatmap_clusters_data(
            self.analyzer.paper_relations_graph, self.analyzer.df, self.analyzer.comp_sizes
        )

        # Find data for comp_x=i and comp_y=j in DataFrame
        def index(i, j):
            return np.logical_and(cluster_edges['comp_x'] == str(i), cluster_edges['comp_y'] == str(j))

        self.assertListEqual(clusters, ['1', '2', '3'], 'Wrong clusters')

        expected_values = np.array([[360, 20, 0],
                                    [20, 233, 0],
                                    [0, 0, 0]])

        n_comps = len(self.analyzer.components)
        for i in range(n_comps):
            for j in range(n_comps):
                self.assertAlmostEqual(cluster_edges[index(i + 1, j + 1)]['value'].values[0],
                                       expected_values[i, j], places=3,
                                       msg=f'Wrong value for comp_x {i} and comp_y {j}')

        expected_densities = np.array([[10.0, 0.833, 0],
                                       [0.833, 14.5625, 0],
                                       [0, 0, 0]])

        n_comps = len(self.analyzer.components)
        for i in range(n_comps):
            for j in range(n_comps):
                self.assertAlmostEqual(cluster_edges[index(i + 1, j + 1)]['density'].values[0],
                                       expected_densities[i, j], places=3,
                                       msg=f'Wrong density for comp_x {i} and comp_y {j}')
