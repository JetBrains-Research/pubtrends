import unittest

import numpy as np
from parameterized import parameterized

from models.keypaper.visualization import Plotter
from models.keypaper.visualization_data import PlotPreprocessor
from models.test.mock_analyzer import MockAnalyzer


class TestPlotPreprocessor(unittest.TestCase):

    def setUp(self):
        self.analyzer = MockAnalyzer()
        self.plotter = Plotter(self.analyzer)

    @parameterized.expand([
        ('#91C82F', [145, 200, 47]),
        ('#8ffe09', [143, 254, 9])
    ])
    def test_hex2rgb(self, color, expected):
        self.assertEqual(PlotPreprocessor.hex2rgb(color), expected)

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
        comps, source = PlotPreprocessor.component_ratio_data(
            self.analyzer.df, self.plotter.comp_palette
        )

        expected_comps = ['3', '2', '1']
        expected_ratios = [9.090909, 36.363636, 54.545454]
        expected_colors = list(reversed(self.plotter.comp_palette))

        self.assertEqual(comps, expected_comps, 'Wrong list of components')
        for ratio, expected in zip(source.data['ratios'], expected_ratios):
            self.assertAlmostEqual(ratio, expected, places=3, msg='Wrong component ratio')
        self.assertEqual(source.data['colors'], expected_colors, 'Wrong list of component colors')

    def test_paper_statistics_data(self):
        ds = PlotPreprocessor.papers_statistics_data(self.analyzer.df)

        expected_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        expected_counts = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 6, 2, 2]

        self.assertEqual(list(ds.data['year']), expected_years, 'Wrong list of years')
        self.assertEqual(list(ds.data['counts']), expected_counts, 'Wrong list of paper counts')

    def test_chord_diagram_data(self):
        layout, node_data_source, edge_data_source = PlotPreprocessor.chord_diagram_data(
            self.analyzer.CG, self.analyzer.df, self.analyzer.pm, self.analyzer.comp_other, self.plotter.comp_palette
        )

        expected_nodes = self.analyzer.CG.nodes()
        expected_edges = self.analyzer.CG.edges()
        data = node_data_source.data

        self.assertCountEqual(list(layout.keys()), expected_nodes, 'Wrong nodes in layout')
        self.assertCountEqual(list(data['index']), expected_nodes,
                              'Wrong order of nodes in data source')

        # Check that the size of all nodes is defined and is positive
        self.assertFalse(np.any(np.isnan(data['size'])))
        self.assertTrue(np.all(data['size'] > 0))

        # Check that data corresponds to correct nodes
        for node in expected_nodes:
            idx = list(data['index']).index(node)
            comp = self.analyzer.df[self.analyzer.df['id'] == node]['comp'].values[0]
            expected_topic = f'#{comp + 1} OTHER' if comp == self.analyzer.comp_other else f'#{comp + 1}'

            self.assertEqual(data['colors'][idx], self.plotter.comp_palette[comp], f'Wrong color for node {node}')
            self.assertEqual(data['topic'][idx], expected_topic, f'Wrong topic for node {node}')

    def test_chord_diagram_layout(self):
        nodes = self.analyzer.CG.nodes()
        edges = self.analyzer.CG.edges()

        layout, xs, ys = PlotPreprocessor.chord_diagram_layout(nodes, edges)

        self.assertEqual(len(layout), len(nodes), 'Wrong number of nodes in layout')
        self.assertCountEqual(layout.keys(), nodes, 'Wrong nodes in layout')

        edges_found = 0
        inverse_layout = {pos: node for node, pos in layout.items()}
        for x, y in zip(xs, ys):
            start_node = inverse_layout[(x[0], y[0])]
            end_node = inverse_layout[(x[-1], y[-1])]
            self.assertIn((start_node, end_node), list(edges), 'Wrong edge in layout')
            edges_found += 1

        self.assertEqual(edges_found, len(edges), 'Some edges from co-citation graph are missing')

    def test_article_view_data_source(self):
        width = 760
        ds = PlotPreprocessor.article_view_data_source(
            self.analyzer.df, self.analyzer.min_year, self.analyzer.max_year, width=width
        )

        self.assertFalse(np.any(ds.data['year'] == np.nan), 'NaN values in `year` column')
        self.assertFalse(np.any(ds.data['size'] == np.nan), 'NaN values in `size` column')

        max_size = np.max(ds.data['size'])
        max_width = max_size * (self.analyzer.max_year - self.analyzer.min_year + 1)
        self.assertLessEqual(max_width, width, 'Horizontal overlap')

    def test_heatmap_clusters(self):

        # Find data for comp_x=i and comp_y=j in DataFrame
        def cell_filter(i, j):
            return np.logical_and(cluster_edges['comp_x'] == str(i), cluster_edges['comp_y'] == str(j))

        cluster_edges, clusters = PlotPreprocessor.heatmap_clusters_data(
            self.analyzer.CG, self.analyzer.df, self.analyzer.pmcomp_sizes
        )

        self.assertListEqual(clusters, ['1', '2', '3'], 'Wrong clusters')

        expected_values = np.array([[36, 2, 0],
                                    [2, 26, 0],
                                    [0, 0, 0]])

        n_comps = len(self.analyzer.components)
        for i in range(n_comps):
            for j in range(n_comps):
                self.assertAlmostEqual(cluster_edges[cell_filter(i + 1, j + 1)]['value'].values[0],
                                       expected_values[i, j], places=3,
                                       msg=f'Wrong value for comp_x {i} and comp_y {j}')

        expected_densities = np.array([[1.0, 0.083, 0],
                                       [0.083, 1.625, 0],
                                       [0, 0, 0]])

        n_comps = len(self.analyzer.components)
        for i in range(n_comps):
            for j in range(n_comps):
                self.assertAlmostEqual(cluster_edges[cell_filter(i + 1, j + 1)]['density'].values[0],
                                       expected_densities[i, j], places=3,
                                       msg=f'Wrong density for comp_x {i} and comp_y {j}')

    def test_subtopic_evolution_data(self):
        edges, nodes_data = PlotPreprocessor.subtopic_evolution_data(
            self.analyzer.evolution_df, self.analyzer.evolution_kwds, self.analyzer.n_steps
        )

        expected_edges = [('2014 -1', '2019 0', 1), ('2014 -1', '2019 1', 4),
                          ('2014 0', '2019 0', 3), ('2014 1', '2019 1', 2)]
        expected_nodes_data = [('2019 0', '2019 shiftwork, estrogen, pattern, disturbance, cell'),
                               ('2019 1', '2019 study, analysis, association, time, cpg'),
                               ('2014 -1', 'TBD'),
                               ('2014 0', '2014 body, susceptibility, ieaa, risk, time'),
                               ('2014 1', '2014 reaction, disturbance, pattern, study, rhythm')]

        self.assertCountEqual(edges, expected_edges, 'Wrong Sankey diagram edges')
        self.assertListEqual([el[0] for el in nodes_data], [el[0] for el in expected_nodes_data],
                             'Wrong node order')
        self.assertListEqual(nodes_data, expected_nodes_data, 'Wrong nodes data')

    def test_subtopic_evolution_keywords(self):
        _, source = PlotPreprocessor.subtopic_evolution_keywords_data(
            self.analyzer.evolution_kwds
        )

        expected_keywords_data = {
            'years': [2014, 2014, 2019, 2019],
            'subtopics': [1, 2, 1, 2],
            'keywords': [
                'body, susceptibility, ieaa, risk, time, acceleration, gene, association, tumor, ageaccel, '
                'development, tissue, blood, study, age',
                'reaction, disturbance, pattern, study, rhythm, result, change, analysis, shiftwork, disruption, '
                'per2, per1, promoter, expression, gene',
                'shiftwork, estrogen, pattern, disturbance, cell, per2, disruption, night, analysis, study, rhythm, '
                'per1, promoter, expression, gene',
                'study, analysis, association, time, cpg, sample, development, ageaccel, type, blood, cell, '
                'acceleration, risk, tissue, age'
            ]
        }

        self.assertEqual(expected_keywords_data['years'], source.data['years'], 'Wrong years')
        self.assertEqual(expected_keywords_data['subtopics'], source.data['subtopics'], 'Wrong subtopics')
        self.assertEqual(expected_keywords_data['keywords'], source.data['keywords'], 'Wrong keywords')
