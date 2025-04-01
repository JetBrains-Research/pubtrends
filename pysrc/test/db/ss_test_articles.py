import pandas as pd

from pysrc.papers.db.ss_article import SemanticScholarArticle

article1 = SemanticScholarArticle(ssid='5451b1ef43678d473575bdfa7016d024146f2b53', crc32id=-410264312, pmid=None,
                                  title='I can find this using full text search',
                                  year=1999, doi='10.000/0000')

article2 = SemanticScholarArticle(ssid='cad767094c2c4fff5206793fd8674a10e7fba3fe', crc32id=984465402, pmid=None,
                                  title='Can find using search.', abstract='Abstract 1',
                                  year=1980)

article3 = SemanticScholarArticle(ssid='e7cdbddc7af4b6138227139d714df28e2090bd5f', crc32id=17079054, pmid=None,
                                  title='Use search to find it')

article4 = SemanticScholarArticle(ssid='3cf82f53a52867aaade081324dff65dd35b5b7eb', crc32id=-1875049083, pmid=None,
                                  title='Want to find it? Just search', year=1976)

article5 = SemanticScholarArticle(ssid='5a63b4199bb58992882b0bf60bc1b1b3f392e5a5', crc32id=1831680518, pmid=1,
                                  title='Search is key to find', abstract='Abstract 4',
                                  year=2003)

article6 = SemanticScholarArticle(ssid='7dc6f2c387193925d3be92d4cc31c7a7cea66d4e', crc32id=-1626578460, pmid=2,
                                  title='Article 6 is here', abstract='Abstract 6',
                                  year=2018)

article7 = SemanticScholarArticle(ssid='0f9c1d2a70608d36ad7588d3d93ef261d1ae3203', crc32id=1075821748, pmid=3,
                                  title='Article 7 is here', abstract='Abstract 7',
                                  year=2010)

article8 = SemanticScholarArticle(ssid='872ad0e120b9eefd334562149c065afcfbf90268', crc32id=-1861977375, pmid=4,
                                  title='Article 8 is here', abstract='Abstract 8',
                                  year=1937)

article9 = SemanticScholarArticle(ssid='89ffce2b5da6669f63c99ff6398b312389c357dc', crc32id=-1190899769, pmid=5,
                                  title='Article 9 is here', abstract='Abstract 9')

article10 = SemanticScholarArticle(ssid='390f6fbb1f25bfbc53232e8248c581cdcc1fb9e9', crc32id=-751585733, pmid=6,
                                   title='Article 10 is here', abstract='Abstract 10',
                                   year=2017)

REQUIRED_ARTICLES = [article1, article2, article3, article4, article6, article7, article8, article9, article10]
EXTRA_ARTICLES = [article5]
REQUIRED_CITATIONS = [(article1, article4), (article1, article3), (article1, article8),
                      (article3, article8), (article2, article4), (article2, article3),
                      (article6, article7), (article6, article10)]
EXTRA_CITATIONS = [(article5, article1)]

CITATIONS_STATS = [[article1.ssid, article5.year, 1],
                   [article3.ssid, article1.year, 1],
                   [article3.ssid, article2.year, 1],
                   [article4.ssid, article1.year, 1],
                   [article4.ssid, article2.year, 1],
                   [article7.ssid, article6.year, 1],
                   [article8.ssid, article1.year, 1],
                   [article8.ssid, article3.year, 1],
                   [article10.ssid, article6.year, 1]]

EXPECTED_CIT_STATS_DF = pd.DataFrame(CITATIONS_STATS, columns=['id', 'year', 'count']) \
    .sort_values(by=['id', 'year']).reset_index(drop=True)

PUB_DF = pd.DataFrame.from_records([article.to_dict() for article in REQUIRED_ARTICLES])
PUB_DF.abstract = ''

EXPECTED_CIT_DF = pd.DataFrame([(article_out.ssid, article_in.ssid) for article_out, article_in in REQUIRED_CITATIONS],
                               columns=['id_out', 'id_in']).sort_values(by=['id_out', 'id_in']).reset_index(drop=True)

COCITATIONS_DATA = [[article6.ssid, article7.ssid, article10.ssid, article6.year],
                    [article1.ssid, article4.ssid, article3.ssid, article1.year],
                    [article2.ssid, article4.ssid, article3.ssid, article2.year],
                    [article1.ssid, article8.ssid, article3.ssid, article1.year],
                    [article1.ssid, article4.ssid, article8.ssid, article1.year]]

EXPECTED_COCIT_DF = pd.DataFrame(COCITATIONS_DATA, columns=['citing', 'cited_1', 'cited_2', 'year']) \
    .sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)

ARTICLES_LIST = [article1, article4, article3, article8, article7, article10]
EXPANDED_ARTICLES_DF = pd.DataFrame(
    [[article5.ssid, 1], [article6.ssid, 1], [article2.ssid, 1]], columns=['id', 'total']
)