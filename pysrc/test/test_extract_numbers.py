import unittest

from parameterized import parameterized

from pysrc.papers.extract_numbers import extract_metrics


class TestExtractNumbers(unittest.TestCase):

    @parameterized.expand([
        ("No specific antiviral drug has been proven effective for treatment of patients with"
         " severe coronavirus disease 2019 (COVID-19).",
         {'disease': [(2019, 0)]}),

        ("Over brief training periods of 3-24 min, four patients then used these signals"
         " to master closed-loop control and to achieve success rates of 74-100% in a"
         " one-dimensional binary task.",
         {'min': [(3, 0), (24, 0)], 'patient': [(4, 0)], 'rate': [(74, 0), (100, 0)]}),

        ("We performed a weighted multivariate analysis of urinary creatinine concentrations in"
         " 22,245 participants of the third National Health and Nutrition Examination Survey (1988-1994)"
         " and established reference ranges (10th-90th percentiles) for each demographic and age category.",
         {'participant': [(22245, 0)],
          'percentile': [(10, 0), (90, 0)],
          'survey': [(1988, 0), (1994, 0)]}),

        ("Longitudinal descriptive analyses of the 1032 participants in the 1991-2007"
         " National Institute of Child Health and Human Development Study of Early Child Care and"
         " Youth Development birth cohort from 10 study sites who had accelerometer-determined minutes of MVPA"
         " at ages 9 (year 2000), 11 (2002), 12 (2003), and 15 (2006) years.",
         {'age': [(9, 0)],
          'analysis': [(11, 0), (2002, 0), (12, 0), (2003, 0)],
          'institute': [(1991, 0), (2007, 0)],
          'participant': [(1032, 0)],
          'site': [(10, 0)],
          'year': [(2000, 0), (15, 0), (2006, 0)]}),

        ("Hookworm infection occurs in almost half of ssa's poorest people, including 40-50 million school-aged"
         " children and 70 million pregnant women in whom it is a leading cause of anemia.",
         {'child': [(40000000, 0), (50000000, 0)], 'woman': [(70000000, 0)]}),

        ("For the 2 most mutagenic regimens: 4 x 1 hr in 3 mm enu and 6 x 1', 5: 'hr in 3 mm enu'.",
         {'enu': [(6, 0), (1, 0)],
          'hr': [(4, 0), (1, 0), (5, 0)],
          'mm': [(3, 0), (3, 0)],
          'regimen': [(2, 0)]}),

        ("Multivariate analysis showed that age (odds ratio [OR], 1.06; 95% confidence interval [CI], 1.02 - 1.11;"
         " p = 0.004), presence of mechanical ventilation (OR, 11.1; 95% CI, 1.92 - 63.3; p = 0.007) and"
         " fluoroquinolone exposure during hospitalization (OR, 28.9; 95% CI, 1.85 - 454.6; p = 0.02) were"
         " independent risk factors for KPC in patients with K. pneumoniae bacteremia.",
         {'ci': [(95, 1), (1.92, 1), (63.3, 1), (1.85, 2), (454.6, 2)],
          'hospitalization': [(28.9, 2), (95, 2)],
          'interval': [(95, 1)],
          'p': [(0.007, 2), (0.02, 2)],
          'ratio': [(1.06, 1), (1.02, 1), (1.11, 1), (0.004, 1)],
          'ventilation': [(11.1, 1)]}),

        ("All patients with SBP had polymorphonuclear cell count in ascitic fluid > 250/mm(3).",
         {'mm': [(250, 1), (3, 1)]}),
    ])
    def test_numbers(self, sentence, numbers):
        metrics, _ = extract_metrics(sentence, visualize_dependencies=False)
        self.assertEqual(numbers, metrics)

    TEXT = """
Two bananas.
10 million USD.
Twenty million dogs.
    """

    def test_sentences(self):
        # One is missing here!
        metrics, _ = extract_metrics(TestExtractNumbers.TEXT, visualize_dependencies=False)
        self.assertEqual({'banana': [(2, 0)], 'dog': [(20000000, 2)], 'usd': [(10000000, 1)]}, metrics)

    def test_missing_one(self):  # Missing one
        metrics, _ = extract_metrics("One apple", visualize_dependencies=False)
        self.assertEqual({}, metrics)


if __name__ == '__main__':
    unittest.main()
