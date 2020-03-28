import unittest

from models.keypaper.plot_preprocessor_experimental import ExperimentalPlotPreprocessor
from models.keypaper.plotter_experimental import ExperimentalPlotter
from models.test.mock_analyzer import MockAnalyzer


class TestExperimentalPlotPreprocessor(unittest.TestCase):

    def setUp(self):
        self.analyzer = MockAnalyzer()
        self.plotter = ExperimentalPlotter(self.analyzer)

    def test_subtopic_evolution_data(self):
        edges, nodes_data = ExperimentalPlotPreprocessor.subtopic_evolution_data(
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
        _, source = ExperimentalPlotPreprocessor.subtopic_evolution_keywords_data(
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
