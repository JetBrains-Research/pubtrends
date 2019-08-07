from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass
class Article:
    ssid: str
    crc32id: int
    title: str
    year: int = None

    def to_db_publication(self):
        return "('{0}', {1}, '{2}', {3})".format(self.ssid, self.crc32id, self.title,
                                                 self.year if self.year else 'null')

    def indexes(self):
        return "('{0}', {1})".format(self.ssid, self.crc32id)

    def to_dict(self):
        return {
            'ssid': self.ssid,
            'crc32id': self.crc32id,
            'title': self.title,
            'year': self.year
        }


article1 = Article('5451b1ef43678d473575bdfa7016d024146f2b53', -410264312,
                   'Differences between Salt-sensitive and Salt-tolerant Genotypes of the Tomato.',
                   year=1999)

article2 = Article('cad767094c2c4fff5206793fd8674a10e7fba3fe', 984465402,
                   'Ear injury and its therapy at the ORL clinic in Olomouc from 1967 to 1976.',
                   year=1980)

article3 = Article('e7cdbddc7af4b6138227139d714df28e2090bd5f', 17079054,
                   'Laser-based sensing for assessing and monitoring civil infrastructures')

article4 = Article('3cf82f53a52867aaade081324dff65dd35b5b7eb', -1875049083,
                   'Multiwavelength shearography for quantitative measurements '
                   'of two-dimensional strain distributions.',
                   year=1976)

article5 = Article('5a63b4199bb58992882b0bf60bc1b1b3f392e5a5', 1831680518,
                   'The Problem of Safe Milk.',
                   year=2003)

article6 = Article('7dc6f2c387193925d3be92d4cc31c7a7cea66d4e', -1626578460,
                   'Retrospective analysis of efficacy and safety of amrubicin in refractory '
                   'and relapsed small-cell lung cancer',
                   year=2018)

article7 = Article('0f9c1d2a70608d36ad7588d3d93ef261d1ae3203', 1075821748,
                   'The inside view on plant growth',
                   year=2010)

article8 = Article('872ad0e120b9eefd334562149c065afcfbf90268', -1861977375,
                   'Crystal structure of the Escherichia coli Tas protein, '
                   'an NADP(H)-dependent aldo-keto reductase.',
                   year=1937)

article9 = Article('89ffce2b5da6669f63c99ff6398b312389c357dc', -1190899769,
                   'Ischiopubic synchondrosis as a case of non specific groin pain '
                   'in a 12 year old football player')

article10 = Article('390f6fbb1f25bfbc53232e8248c581cdcc1fb9e9', -751585733,
                    'Lost wax-bolus technique to process closed hollow obturator '
                    'with uniform wall thickness using single flasking procedure',
                    year=2017)

required_articles = [article1, article2, article3, article4, article6, article7, article8, article9,
                     article10]
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
                   [article10.ssid, article6.year, 1]]

cit_stats_df = pd.DataFrame(citations_stats, columns=['id', 'year', 'count']).sort_values(
    by=['id', 'year']
).reset_index(drop=True)

pub_df = pd.DataFrame.from_records([article.to_dict() for article in required_articles])
pub_df.abstract = ''

cit_df = pd.DataFrame([(article_out.ssid, article_in.ssid) for article_out, article_in in required_citations],
                      columns=['id_out', 'id_in'])

expected_graph = nx.DiGraph()
for citation in required_citations:
    u, v = citation
    expected_graph.add_edge(u.ssid, v.ssid)

expected_cgraph = nx.Graph()
expected_cgraph.add_weighted_edges_from([(article7.ssid, article10.ssid, 1),
                                         (article4.ssid, article3.ssid, 2),
                                         (article3.ssid, article8.ssid, 1),
                                         (article4.ssid, article8.ssid, 1)])

cocitations = [[article6.ssid, article7.ssid, article10.ssid, article6.year],
               [article1.ssid, article4.ssid, article3.ssid, article1.year],
               [article2.ssid, article4.ssid, article3.ssid, article2.year],
               [article1.ssid, article3.ssid, article8.ssid, article1.year],
               [article1.ssid, article4.ssid, article8.ssid, article1.year]]

cocitations_df = pd.DataFrame(cocitations, columns=['citing', 'cited_1', 'cited_2', 'year'])
cocitations_df = cocitations_df.sort_values(by=['citing', 'cited_1', 'cited_2']).reset_index(drop=True)

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
