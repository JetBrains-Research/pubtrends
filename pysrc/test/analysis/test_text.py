import unittest

import numpy as np

from pysrc.config import PubtrendsConfig
from pysrc.papers.analysis.text import stemmed_tokens, _build_stems_to_tokens_map
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.utils import SORT_MOST_CITED
from pysrc.test.mock_loaders import MockLoader

PUBTRENDS_CONFIG = PubtrendsConfig(test=True)


class TestText(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        analyzer = PapersAnalyzer(MockLoader(), PUBTRENDS_CONFIG, test=True)
        ids = analyzer.search_terms(query='query')
        analyzer.analyze_papers(ids, PUBTRENDS_CONFIG.show_topics_default_value, test=True)
        cls.data = analyzer.save(None, 'query', 'Pubmed', SORT_MOST_CITED, 10, False, None, None)

    def test_stemmed_tokens(self):
        text = """Very interesting article about elephants and donkeys.
        There are two types of elephants - Indian and African.
        Both of them are really beautiful, but in my opinion Indian are even cuter"""
        # nouns and adjectives from text excluding comparative and superlative forms
        expected = [('interest', 'interesting'), ('articl', 'article'), ('eleph', 'elephant'), ('donkey', 'donkey'),
                    ('type', 'type'), ('eleph', 'elephant'), ('indian', 'indian'), ('african', 'african'),
                    ('realli', 'really'), ('beauti', 'beautiful'), ('opinion', 'opinion'), ('indian', 'indian'),
                    ('even', 'even'), ('cute', 'cute')]
        actual = stemmed_tokens(text)
        # print(actual)
        self.assertSequenceEqual(actual, expected)

    def test_stemmed_tokens_map(self):
        text = """Control differences in Bland-Altman plots for mean+/-SD in mm Hg were systolic, 0.0+/-4.4; diastolic, 
         0.6+/-1.7; pulse, -0.7+/-4.2; and mean pressure, -0.5+/-2.0. For nitroglycerin infusion, differences 
         respectively were systolic, -0.2+/-4.3; diastolic, 0.6+/-1.7; pulse, -0.8+/-4.1; and mean pressure, 
         -0.4+/-1.8. Differences were within specified limits of the "Association for the Advancement of Medical 
         Instrumentation" SP10 criteria. In contrast, differences between recorded radial and aortic systolic and pulse 
         pressures were well outside the criteria (respectively, 15.7+/-8.4 and 16.3+/-8.5 for control and 14.5+/-7.3 
         and 15.1+/-7.3 mm Hg for nitroglycerin) 
         ...
         One particular BCI approach is the so-called 'P300 matrix speller' that was first described by 
         Farwell and Donchin (1988 Electroencephalogr. Clin. Neurophysiol. 70 510-23). ... it 
         relies primarily on the P300-evoked potential and minimally, if at all, on other EEG features such as the 
         visual-evoked potential (VEP).  ... We evaluated the performance of 
         17 healthy subjects using a 'P300' matrix speller under two conditions. Under one condition ('letter'), the 
         subjects focused their eye gaze on the intended letter, while under the second condition ('center'), the 
         subjects focused their eye gaze on a fixation cross that was located in the center of the matrix ...
         The applicability of these findings to people with severe neuromuscular disabilities 
         (particularly in eye-movements) remains to be determined.
         """
        expected = {'differ': 'difference', 'systol': 'systolic', 'diastol': 'diastolic', 'puls': 'pulse',
                    'pressur': 'pressure', 'infus': 'infusion', 'respect': 'respectively', 'specifi': 'specify',
                    'associ': 'association', 'advanc': 'advancement', 'medic': 'medical',
                    'instrument': 'instrumentation', 'so-cal': 'so-called', 'describ': 'describe', 'farwel': 'farwell',
                    'reli': 'rely', 'primarili': 'primarily', 'p300-evok': 'p300-evoked', 'potenti': 'potential',
                    'minim': 'minimally', 'featur': 'feature', 'visual-evok': 'visual-evoked', 'evalu': 'evaluate',
                    'perform': 'performance', 'healthi': 'healthy', 'condit': 'condition', 'intend': 'intended',
                    'fixat': 'fixation', 'locat': 'locate', 'applic': 'applicability', 'find': 'finding',
                    'peopl': 'people', 'sever': 'severe', 'disabl': 'disability', 'particular': 'particularly',
                    'eye-mov': 'eye-movements', 'remain': 'remains', 'determin': 'determine'}
        actual = _build_stems_to_tokens_map(stemmed_tokens(text))
        print(actual)
        self.assertSequenceEqual(actual, expected)

    def test_dashes(self):
        text = """Genome-wide whole-genome DNA-methylation"""
        expected = [('genome-wid', 'genome-wide'), ('whole-genom', 'whole-genome'), ('dna-methyl', 'dna-methylation')]
        actual = stemmed_tokens(text)
        # print(actual)
        self.assertSequenceEqual(actual, expected)

    def test_corpus_vectorization(self):
        self.assertEqual(
            self.data.corpus_tokens,
            ['abstract',
             'article',
             'breakthrough',
             'interesting',
             'paper',
             'term1',
             'term2',
             'term3',
             'term4',
             'term5']
        )
        self.assertTrue(np.array_equal(
            self.data.corpus_counts.toarray(),
            [[0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
             [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
             [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
             [0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]]))
