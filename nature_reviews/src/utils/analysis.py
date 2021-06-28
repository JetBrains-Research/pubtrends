import numpy as np
import pandas as pd

from pysrc.papers.analysis.graph import node2vec
from pysrc.papers.analysis.topics import cluster_embeddings
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import AnalyzerSettings


def get_direct_references_subgraph(analyzer, pmid):
    """
    Extract subgraph of the similarity graph containing only direct references
    of the paper with given `pmid`.
    """
    references = list(analyzer.citations_graph.successors(pmid))
    references.append(pmid)

    return analyzer.similarity_graph.subgraph(references)


def get_similarity_func(similarity_bibliographic_coupling, similarity_cocitation,
                        similarity_citation, similarity_text_citation):
    def inner(d):
        return similarity_bibliographic_coupling * np.log1p(d.get('bibcoupling', 0)) + \
               similarity_cocitation * np.log1p(d.get('cocitation', 0)) + \
               similarity_citation * d.get('citation', 0) + \
               similarity_text_citation * d.get('text', 0)

    return inner


def topic_analysis_node2vec(analyzer, subgraph, **settings):
    """
    Rerun topic analysis based on node2vec for a given similarity graph and settings.
    """
    node_ids, node_embeddings = node2vec(subgraph,
                                         weight_func=PapersAnalyzer.similarity,
                                         walk_length=settings['walk_length'], 
                                         walks_per_node=settings['walks_per_node'], 
                                         vector_size=settings['vector_size'])
    clusters, _ = cluster_embeddings(
        node_embeddings, settings['topic_min_size'], settings['topics_max_number']
    )
    return dict(zip(node_ids, clusters))


def rebuild_similarity_graph(analyzer, min_cocitation=0):
    """
    Rebuild similarity graph only with edges that have cocitation > min_cocitations
    """
    cocit_data = []
    bibcoupling_data = []
    for cited_1, cited_2, data in analyzer.similarity_graph.edges(data=True):
        if 'cocitation' in data and data['cocitation'] > min_cocitation:
            cocit_data.append((cited_1, cited_2, data['cocitation']))
        if 'bibcoupling' in data:
            bibcoupling_data.append((cited_1, cited_2, data['bibcoupling']))

    cocit_df = pd.DataFrame(cocit_data, columns=['cited_1', 'cited_2', 'total'])
    bibcoupling_df = pd.DataFrame(bibcoupling_data, columns=['citing_1', 'citing_2', 'total'])

    analyzer.similarity_graph = build_similarity_graph(
        analyzer.df, analyzer.texts_similarity,
        analyzer.citations_graph, cocit_df, bibcoupling_df
    )


def align_clusterings_for_sklearn(partition, ground_truth):
    # Get clustering subset only with IDs present in ground truth dict
    actual_clustering = {k: v for k, v in partition.items() if k in ground_truth}

    # Align clusterings
    labels_true = []
    labels_pred = []

    for pmid in actual_clustering:
        labels_true.append(ground_truth[pmid])
        labels_pred.append(actual_clustering[pmid])

    return labels_true, labels_pred
