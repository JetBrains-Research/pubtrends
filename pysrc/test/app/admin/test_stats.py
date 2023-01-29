import unittest

from parameterized import parameterized

from pysrc.app.admin.stats import duration_seconds, parse_stats_content

# Part of preprocessed app.log file
LOG = """
[2021-10-07 04:34:42,801: INFO] /search_paper addr:172.30.2.189 args:{}
[2021-10-07 04:34:42,961: INFO] /process paper analysis addr:172.30.2.189 args:\
    {"query": "Paper title", "analysis_type": "paper", "key": "title", \
    "value": "On the features of ...", "source": "Pubmed", "jobid": "400a5897"}
[2021-10-07 04:34:49,162: INFO] /status failure. Search error: True. addr:172.30.2.189 args:{"jobid": "400a5897"}
 "programming languages theory"}

[2021-10-06 14:06:57,517: INFO] /process regular search addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "jobid": "fce96ec4"}
[2021-10-06 20:09:13,591: INFO] /result success addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "jobid": "fce96ec4"}
[2021-10-06 20:09:38,888: INFO/pysrc.review.app.app] /generate_review addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "fce96ec4", "papers_number": "10", "sents_number": "1"}
[2021-10-06 20:09:52,168: INFO/pysrc.review.app.app] /review addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "jobid": "49521028"}
[2021-10-06 20:10:51,608: INFO] /graph success similarity addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "jobid": "fce96ec4"}
[2021-10-06 20:10:58,608: INFO] /papers success addr:172.30.0.150 args:\
    {"query": "human microbiome", "source": "Pubmed", "comp": "1", "jobid": "fce96ec4"}
[2023-01-28 22:48:31,871: INFO/pysrc.app.app] /paper success addr:172.18.0.1 args:\
    {"query": "human microbiome", "source": "Pubmed", "comp": "1", "jobid": "fce96ec4"}
[2023-01-28 22:17:25,126: INFO/pysrc.app.app] /process review addr:172.18.0.1 args:\
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

EXPECTED_STATS = {
    'paper_searches_avg_duration': '00:03:59',
    'paper_searches_recent': [('2021-10-12 09:12:49',
                               'Semantic Scholar',
                               'Paper doi=10.1063/5.0021420',
                               '/paper?query=Paper '
                               'doi=10.1063/5.0021420&analysis_type=paper&source=Semantic '
                               'Scholar&key=doi&value=10.1063/5.0021420&jobid=96584252',
                               '00:03:59',
                               'Ok'),
                              ('2021-10-07 04:34:42',
                               'Pubmed',
                               'Paper title',
                               '/paper?query=Paper '
                               'title&analysis_type=paper&key=title&value=On the '
                               'features of ...&source=Pubmed&jobid=400a5897',
                               '-',
                               'N/A')],
    'paper_searches_successful': 1,
    'paper_searches_total': 2,
    'recent': 50,
    'searches_graph_shown': 1,
    'searches_papers_clicks': 1,
    'searches_papers_list_shown': 1,
    'searches_review_shown': 1,
    'terms_searches_avg_duration': '03:03:59',
    'terms_searches_recent': [('2021-10-07 14:46:24',
                               'Semantic Scholar',
                               'Bla-bla-bla',
                               '/result?query=Bla-bla-bla&source=Semantic '
                               'Scholar&jobid=23818aa8',
                               '00:05:42',
                               'Not found',
                               0,
                               '-',
                               '-',
                               '-'),
                              ('2021-10-06 14:06:57',
                               'Pubmed',
                               'human microbiome',
                               '/result?query=human '
                               'microbiome&source=Pubmed&jobid=fce96ec4',
                               '06:02:16',
                               'Ok',
                               1,
                               '+',
                               '+',
                               '+')],
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
