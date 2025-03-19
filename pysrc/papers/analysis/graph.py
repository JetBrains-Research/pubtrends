import logging

import networkx as nx
import numpy as np

from pysrc.papers.config import *

logger = logging.getLogger(__name__)


def similarity(d):
    return \
            SIMILARITY_BIBLIOGRAPHIC_COUPLING * d.get('bibcoupling', 0) + \
            SIMILARITY_COCITATION * d.get('cocitation', 0) + \
            SIMILARITY_CITATION * d.get('citation', 0)


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


def sparse_graph(graph, k, key='similarity', add_similarity=True):
    logger.debug(f'Building {k}-neighbours sparse graph, '
                 f'edges/nodes={graph.number_of_edges() / graph.number_of_nodes()}')
    if add_similarity:
        for i, j in graph.edges():
            graph[i][j]['similarity'] = similarity(graph.get_edge_data(i, j))
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


def add_artificial_text_similarities_edges(ids, texts_embeddings, sparse_papers_graph):
    for node_i, node in enumerate(ids):
        # Compute distances from the current node to all other nodes
        distances = np.linalg.norm(texts_embeddings - texts_embeddings[node_i], axis=1)
        similarities = 1 / (1 + distances)
        # Get indices of closest nodes (excluding the source node itself)
        for similar_node_i in np.argsort(similarities)[-GRAPH_TEXT_SIMILARITY_EDGES-1:-1]:
            similar_node = ids[similar_node_i]
            if not sparse_papers_graph.has_edge(node, similar_node):
                sparse_papers_graph.add_edge(node, similar_node, textsimilarity=(similarities[similar_node_i]))
        # Add text_similarity to all existing edges
        for neighbor_i, neighbor in enumerate(ids):
            if neighbor_i > node_i and sparse_papers_graph.has_edge(node, neighbor):
                sparse_papers_graph[node][neighbor]['textsimilarity'] = similarities[neighbor_i]
