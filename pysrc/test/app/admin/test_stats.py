import unittest

from parameterized import parameterized

from pysrc.app.admin.stats import duration_seconds, parse_stats_content

# Part of preprocessed app.log file
LOG = """
[2025-12-01 16:55:28,882: INFO/pysrc.app.pubtrends_app] /search_semantic addr:172.29.41.49 args:{} form:{"query": "dna methylation clock", "source": "Pubmed", "limit": "1000", "noreviews": "on", "topics": "10"}
[2025-12-01 16:58:10,361: INFO/pysrc.app.pubtrends_app] /status success addr:172.29.41.49 args:{}
[2025-12-01 16:58:10,399: INFO/pysrc.app.pubtrends_app] /result addr:172.29.41.49 args:{"query": "dna methylation clock", "jobid": "3fb4af6e-257c-4bc6-ba22-8f5cca0a7ca8"}
[2025-12-01 16:58:22,763: INFO/pysrc.app.pubtrends_app] /result success addr:172.29.41.49 args:{"query": "dna methylation clock", "jobid": "3fb4af6e-257c-4bc6-ba22-8f5cca0a7ca8"}
[2025-12-01 16:59:06,547: INFO/pysrc.app.pubtrends_app] /paper addr:172.29.41.49 args:{"query": "dna methylation clock", "id": "24138928", "jobid": "3fb4af6e-257c-4bc6-ba22-8f5cca0a7ca8"}
[2025-12-01 16:59:08,235: INFO/pysrc.app.pubtrends_app] /paper success addr:172.29.41.49 args:{"query": "dna methylation clock", "id": "24138928", "jobid": "3fb4af6e-257c-4bc6-ba22-8f5cca0a7ca8"}
[2025-12-01 16:59:20,582: INFO/pysrc.app.pubtrends_app] /question addr:172.29.41.49 args:{} form:{}
[2025-12-01 16:59:41,799: INFO/pysrc.app.pubtrends_app] /question success addr:172.29.41.49 args:{} form:{}
[2025-12-01 16:59:53,093: INFO/pysrc.app.pubtrends_app] /graph addr:172.29.41.49 args:{"query": "dna methylation clock", "jobid": "3fb4af6e-257c-4bc6-ba22-8f5cca0a7ca8"}
[2025-12-01 16:59:54,813: INFO/pysrc.app.pubtrends_app] /graph success addr:172.29.41.49 args:{"query": "dna methylation clock", "jobid": "3fb4af6e-257c-4bc6-ba22-8f5cca0a7ca8"}    

[2021-10-07 04:34:42,801: INFO] /search_paper addr:172.30.2.189 args:{}
[2021-10-07 04:34:42,961: INFO] /process paper analysis addr:172.30.2.189 args:\
    {"query": "Paper title", "analysis_type": "paper", "key": "title", \
    "value": "On the features of ...", "source": "Pubmed", "jobid": "400a5897"}
[2021-10-07 04:34:49,162: INFO] /status failure. Search error: True. addr:172.30.2.189 args:{"jobid": "400a5897"}
 "programming languages theory"}

[2021-10-06 14:06:57,517: INFO] /process regular search addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "jobid": "fce96ec4"}
[2021-10-06 20:09:13,591: INFO] /result success addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "jobid": "fce96ec4", "papers_number": "10", "sents_number": "1"}
[2021-10-06 20:10:51,608: INFO] /graph success similarity addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "jobid": "fce96ec4"}
[2021-10-06 20:10:58,608: INFO] /papers success addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "comp": "1", "jobid": "fce96ec4"}
[2023-01-28 22:48:31,871: INFO/pysrc.app.app] /paper success addr:172.18.0.1 args:\
    {"query": "human microbiome", "source": "Pubmed", "comp": "1", "jobid": "fce96ec4"}

[2021-10-07 14:46:24,756: INFO] /process regular search addr:172.30.0.150 args:\
    {"query": "Bla-bla-bla", "source": "Semantic Scholar", "jobid": "23818aa8"}
[2021-10-07 14:52:06,805: INFO] /status failure. Search error: True. addr:172.30.0.150 args:\
    {"jobid": "23818aa8"}

[2021-10-12 09:12:49,896: INFO] /process paper analysis addr:172.30.2.189 args:\
    {"query": "Paper doi=10.1063/5.0021420", "analysis_type": "paper", "source": "Semantic Scholar", \
    "key": "doi", "value": "10.1063/5.0021420", "jobid": "96584252"}
[2021-10-12 09:16:49,515: INFO] /paper success addr:172.30.2.189 args:\
    {"source": "Semantic Scholar", "jobid": "96584252", "key": "doi", "value": "10.1063/5.0021420"}    
"""

EXPECTED_STATS = {'feature_counts': {'Graph': 2, 'Papers': 1, 'Question': 0},
                  'features': ['Papers', 'Graph', 'Question'],
                  'paper_searches_avg_duration': '00:03:59',
                  'paper_searches_recent': [('2021-10-12 09:12:49',
                                             'Semantic Scholar',
                                             'N/A',
                                             '/paper?source=Semantic '
                                             'Scholar&jobid=96584252&key=doi&value=10.1063/5.0021420',
                                             '00:03:59',
                                             'Ok')],
                  'paper_searches_successful': 1,
                  'paper_searches_total': 1,
                  'recent': 50,
                  'searches_papers_clicks': 2,
                  'searches_papers_list_shown': 1,
                  'terms_searches_avg_duration': '06:02:16',
                  'terms_searches_features_results': {'Graph': ['+', '+'],
                                                      'Papers': ['+', '-'],
                                                      'Question': ['-', '-']},
                  'terms_searches_recent': [('2021-10-06 14:06:57',
                                             'Pubmed',
                                             'human microbiome',
                                             '/result?query=human '
                                             'microbiome&source=Pubmed&jobid=fce96ec4&papers_number=10&sents_number=1',
                                             '06:02:16',
                                             'Ok',
                                             1),
                                            ('2025-12-01 16:58:22',
                                             '',
                                             'dna methylation clock',
                                             '/result?query=dna methylation '
                                             'clock&jobid=3fb4af6e-257c-4bc6-ba22-8f5cca0a7ca8',
                                             '-',
                                             'Ok',
                                             1)],
                  'terms_searches_successful': 2,
                  'terms_searches_total': 2}


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

    def test_stats(self):
        content = parse_stats_content(LOG.split('\n'))
        self.assertTrue('terms_searches_plot' in content)
        self.assertTrue('paper_searches_plot' in content)
        self.assertTrue('word_cloud' in content)
        del content['terms_searches_plot']
        del content['paper_searches_plot']
        del content['word_cloud']
        self.assertEqual(EXPECTED_STATS, content)
