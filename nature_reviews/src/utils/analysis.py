import pandas as pd

from pysrc.papers.pubtrends_config import AnalyzerSettings
from pysrc.papers.utils import vectorize_corpus, compute_tfidf

PUB_DF_COLUMNS = ['id', 'title', 'abstract', 'year', 'type', 'keywords', 'mesh', 'doi', 'aux']


def get_direct_references_subgraph(analyzer, pmid):
    """
    Extract subgraph of the similarity graph containing only direct references
    of the paper with given `pmid`.
    """
    references = list(analyzer.citations_graph.successors(pmid))
    references.append(pmid)

    references_similarity_graph = analyzer.similarity_graph.subgraph(references)
    return references_similarity_graph


def recalculate_topic_analysis(analyzer, graph=None, settings=AnalyzerSettings()):
    """
    Rerun topic analysis for a given similarity graph and settings.
    """
    if not graph:
        graph = analyzer.similarity_graph
    topics_dendrogram, partition, comp_other, components, comp_sizes = \
        analyzer.topic_analysis(graph,
                                topic_min_size=settings.TOPIC_MIN_SIZE,
                                max_topics_number=settings.TOPICS_MAX_NUMBER,
                                random_state=settings.SEED,
                                similarity_bibliographic_coupling=settings.SIMILARITY_BIBLIOGRAPHIC_COUPLING,
                                similarity_cocitation=settings.SIMILARITY_COCITATION,
                                similarity_citation=settings.SIMILARITY_CITATION,
                                similarity_text_citation=settings.SIMILARITY_TEXT_CITATION)
    return partition


def rebuild_similarity_graph(analyzer):
    """
    Restores missing data based on the analyzer dump and rebuilds similarity graph applying
    scaling to bibliographic coupling and co-citations.
    """
    analyzer.ids = set(analyzer.df['id'])
    analyzer.n_papers = len(analyzer.ids)
    analyzer.pub_types = list(set(analyzer.df['type']))
    analyzer.query = 'restored from PubTrends export'

    analyzer.pub_df = analyzer.df[PUB_DF_COLUMNS]

    analyzer.components = set(analyzer.df['comp'].unique())
    if -1 in analyzer.components:
        analyzer.components.remove(-1)

    settings = AnalyzerSettings()
    analyzer.corpus_ngrams, analyzer.corpus_counts = \
        vectorize_corpus(analyzer.pub_df,
                         max_features=settings.VECTOR_WORDS, n_gram=settings.VECTOR_NGRAMS,
                         min_df=settings.VECTOR_MIN_DF, max_df=settings.VECTOR_MAX_DF)
    tfidf = compute_tfidf(analyzer.corpus_counts)
    analyzer.texts_similarity = analyzer.analyze_texts_similarity(analyzer.pub_df, tfidf,
                                                                  settings.SIMILARITY_TEXT_MIN,
                                                                  settings.SIMILARITY_TEXT_CITATION_N)

    cocit_data = []
    bibcoupling_data = []
    for cited_1, cited_2, data in analyzer.similarity_graph.edges(data=True):
        if 'cocitation' in data:
            cocit_data.append((cited_1, cited_2, data['cocitation']))
        if 'bibcoupling' in data:
            bibcoupling_data.append((cited_1, cited_2, data['bibcoupling']))

    cocit_df = pd.DataFrame(cocit_data, columns=['cited_1', 'cited_2', 'total'])
    bibcoupling_df = pd.DataFrame(bibcoupling_data, columns=['citing_1', 'citing_2', 'total'])

    analyzer.similarity_graph = analyzer.build_similarity_graph(
        analyzer.df, analyzer.texts_similarity,
        analyzer.citations_graph, cocit_df, bibcoupling_df,
        current=0, task=None
    )


def align_clusterings_for_sklearn(partition, ground_truth):
    # Get clustering
    actual_clustering = {k: v for k, v in partition.items() if k in ground_truth}

    # Align clusterings
    labels_true = []
    labels_pred = []

    for pmid in actual_clustering:
        labels_true.append(ground_truth[pmid])
        labels_pred.append(actual_clustering[pmid])

    return labels_true, labels_pred
