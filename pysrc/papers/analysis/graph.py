import logging

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

from pysrc.config import *

logger = logging.getLogger(__name__)


def similarity(d, use_text=True):
    return \
            SIMILARITY_BIBLIOGRAPHIC_COUPLING * d.get('bibcoupling', 0) + \
            SIMILARITY_COCITATION * d.get('cocitation', 0) + \
            SIMILARITY_CITATION * d.get('citation', 0) + \
            (SIMILARITY_TEXT * d.get('textsimilarity', 0) if use_text else 0)


def build_papers_graph(df, cit_df, cocit_df, bibliographic_coupling_df):
    pids = list(df['id'])

    result = nx.Graph()
    # NOTE: we use nodes id as String to avoid problems str keys in jsonify
    # during graph visualization

    # Co-citations
    for start, end, cocitation in zip(cocit_df['cited_1'], cocit_df['cited_2'], cocit_df['total']):
        result.add_edge(start, end, cocitation=cocitation)

    # Bibliographic coupling
    if len(bibliographic_coupling_df) > 0:
        for start, end, bibcoupling in zip(bibliographic_coupling_df['citing_1'],
                                           bibliographic_coupling_df['citing_2'],
                                           bibliographic_coupling_df['total']):
            if result.has_edge(start, end):
                result[start][end]['bibcoupling'] = bibcoupling
            else:
                result.add_edge(start, end, bibcoupling=bibcoupling)

    # Citations
    for start, end in zip(cit_df['id_out'], cit_df['id_in']):
        if result.has_edge(start, end):
            result[start][end]['citation'] = 1
        else:
            result.add_edge(start, end, citation=1)

    # Ensure all the papers are in the graph
    for pid in pids:
        if not result.has_node(pid):
            result.add_node(pid)

    return result


def sparse_graph(graph, k, key='similarity'):
    """
    Constructs a k-nearest neighbors sparse graph based on the input graph.
    """
    logger.debug(f'Building {k}-neighbours sparse graph, '
                 f'edges/nodes={graph.number_of_edges() / graph.number_of_nodes()}')
    result = nx.Graph()
    # Start from nodes with max number of neighbors
    for n in sorted(graph.nodes(), key=lambda x: len(list(graph.neighbors(x))), reverse=True):
        neighbors_data = sorted(list([(x, graph.get_edge_data(n, x)) for x in graph.neighbors(n)]),
                                key=lambda x: x[1][key], reverse=True)
        for x, data in neighbors_data[:k]:
            if not result.has_edge(n, x) and (not result.has_node(x) or len(list(result.neighbors(x))) < k):
                result.add_edge(n, x, **data)
    # Ensure all the nodes present
    for n in graph.nodes():
        if not result.has_node(n):
            result.add_node(n)
    logger.debug(f'Sparse {k}-neighbours graph edges/nodes={result.number_of_edges() / result.number_of_nodes()}')
    return result


def add_text_similarities_edges(ids, texts_embeddings, papers_graph, top_similar):
    """
    Adds edges based on text similarity to a graph and updates existing edges with
    text similarity values as a cosine similarity between text normalized embeddings.

    """
    texts_embeddings_norm = normalize(texts_embeddings, norm='l2', axis=1)
    cosine_similarity_matrix = 1 - cosine_distances(texts_embeddings_norm)
    for node_i, node in enumerate(ids):
        # Get indices of closest nodes (excluding the source node itself)
        similarities = cosine_similarity_matrix[node_i]
        for similar_node_i in np.argsort(similarities)[-top_similar - 1:-1]:
            similar_node = ids[similar_node_i]
            if not papers_graph.has_edge(node, similar_node):
                papers_graph.add_edge(node, similar_node, textsimilarity=(similarities[similar_node_i]))
        # Add text_similarity to all existing edges
        for neighbor_i, neighbor in enumerate(ids):
            if neighbor_i > node_i and papers_graph.has_edge(node, neighbor):
                papers_graph[node][neighbor]['textsimilarity'] = similarities[neighbor_i]
