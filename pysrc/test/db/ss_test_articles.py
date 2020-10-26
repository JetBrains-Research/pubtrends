import networkx as nx
import pandas as pd

from pysrc.papers.db.ss_article import SemanticScholarArticle

article1 = SemanticScholarArticle(ssid='5451b1ef43678d473575bdfa7016d024146f2b53', crc32id=-410264312, pmid=None,
                                  title='I can find this using full text search',
                                  year=1999, doi='10.000/0000')

article2 = SemanticScholarArticle(ssid='cad767094c2c4fff5206793fd8674a10e7fba3fe', crc32id=984465402, pmid=None,
                                  title='Can find using search', abstract='Abstract 1',
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

required_articles = [article1, article2, article3, article4, article6, article7, article8, article9, article10]
extra_articles = [article5]
required_citations = [(article1, article4), (article1, article3), (article1, article8),
                      (article3, article8), (article2, article4), (article2, article3),
                      (article6, article7), (article6, article10)]
extra_citations = [(article5, article1)]

citations_stats = [[article1.ssid, article5.year, 1],
                   [article3.ssid, article1.year, 1],
                   [article3.ssid, article2.year, 1],
                   [article4.ssid, article1.year, 1],
                   [article4.ssid, article2.year, 1],
                   [article7.ssid, article6.year, 1],
                   [article8.ssid, article1.year, 1],
                   [article8.ssid, article3.year, 1],
                   [article10.ssid, article6.year, 1]]

expected_cit_stats_df = pd.DataFrame(citations_stats, columns=['id', 'year', 'count']) \
    .sort_values(by=['id', 'year']).reset_index(drop=True)

pub_df = pd.DataFrame.from_records([article.to_dict() for article in required_articles])
pub_df.abstract = ''

expected_cit_df = pd.DataFrame([(article_out.ssid, article_in.ssid) for article_out, article_in in required_citations],
                               columns=['id_out', 'id_in']).sort_values(by=['id_out', 'id_in']).reset_index(drop=True)

citations_graph = nx.DiGraph()
for citation in required_citations:
    u, v = citation
    citations_graph.add_edge(u.ssid, v.ssid)

expected_cgraph = nx.Graph()
expected_cgraph.add_weighted_edges_from([(article7.ssid, article10.ssid, 1),
                                         (article4.ssid, article3.ssid, 2),
                                         (article3.ssid, article8.ssid, 1),
                                         (article4.ssid, article8.ssid, 1)])

raw_cocitations = [[article6.ssid, article7.ssid, article10.ssid, article6.year],
                   [article1.ssid, article4.ssid, article3.ssid, article1.year],
                   [article2.ssid, article4.ssid, article3.ssid, article2.year],
                   [article1.ssid, article8.ssid, article3.ssid, article1.year],
                   [article1.ssid, article4.ssid, article8.ssid, article1.year]]

expected_cocit_df = pd.DataFrame(raw_cocitations, columns=['citing', 'cited_1', 'cited_2', 'year']) \
    .sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)

cocitations = [[article7.ssid, article10.ssid, 1],
               [article4.ssid, article3.ssid, 2],
               [article3.ssid, article8.ssid, 1],
               [article4.ssid, article8.ssid, 1]]

cocitations_df = pd.DataFrame(cocitations, columns=['cited_1', 'cited_2', 'total']) \
    .sort_values(by=['cited_1', 'cited_2']).reset_index(drop=True)

bibliographic_coupling_df = \
    pd.DataFrame([], columns=['citing_1', 'citing_2', 'total'])

expected_cocit_and_cit_graph = nx.Graph()
expected_cocit_and_cit_graph.add_weighted_edges_from([(article7.ssid, article10.ssid, 1),
                                                      (article4.ssid, article3.ssid, 2),
                                                      (article3.ssid, article8.ssid, 1.3),
                                                      (article4.ssid, article8.ssid, 1),
                                                      (article6.ssid, article7.ssid, 0.3),
                                                      (article6.ssid, article10.ssid, 0.3),
                                                      (article2.ssid, article3.ssid, 0.3),
                                                      (article2.ssid, article4.ssid, 0.3),
                                                      (article1.ssid, article4.ssid, 0.3),
                                                      (article1.ssid, article3.ssid, 0.3),
                                                      (article1.ssid, article8.ssid, 0.3)])

part_of_articles = [article1, article4, article3, article8, article7, article10]
expanded_articles_df = pd.DataFrame(
    [[article1.ssid, 1], [article6.ssid, 1], [article2.ssid, 1],
     [article4.ssid, 2], [article8.ssid, 2], [article3.ssid, 2]],
    columns=['id', 'total']
)

pub_df_given_ids = pd.DataFrame.from_records([article.to_dict() for article in part_of_articles]) \
    .sort_values(by=['ssid']).reset_index(drop=True)
