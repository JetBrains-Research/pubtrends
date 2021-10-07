import logging
from math import ceil
from queue import PriorityQueue

import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

from pysrc.papers.analysis.node2vec import node2vec

logger = logging.getLogger(__name__)


def build_citation_graph(df, cit_df):
    cg = nx.DiGraph()
    for index, row in cit_df.iterrows():
        v, u = row['id_out'], row['id_in']
        cg.add_edge(v, u)
    # Ensure all the nodes are in the graph
    for node in df['id']:
        if not cg.has_node(node):
            cg.add_node(node)
    return cg


def build_similarity_graph(
        df, citations_graph, cocit_df, bibliographic_coupling_df,
        texts_similarity
):
    """ Similarity graph is built using citation graph and text based methods. """
    pids = list(df['id'])

    sg = nx.Graph()
    # NOTE: we use nodes id as String to avoid problems str keys in jsonify
    # during graph visualization

    # Co-citations
    for el in cocit_df[['cited_1', 'cited_2', 'total']].values:
        start, end, cocitation = str(el[0]), str(el[1]), float(el[2])
        sg.add_edge(start, end, cocitation=cocitation)

    # Bibliographic coupling
    if len(bibliographic_coupling_df) > 0:
        for el in bibliographic_coupling_df[['citing_1', 'citing_2', 'total']].values:
            start, end, bibcoupling = str(el[0]), str(el[1]), float(el[2])
            if sg.has_edge(start, end):
                sg[start][end]['bibcoupling'] = bibcoupling
            else:
                sg.add_edge(start, end, bibcoupling=bibcoupling)

    # Text similarity
    if len(df) >= 2:
        for i, pid1 in enumerate(df['id']):
            for j, cos_similarity in texts_similarity[i]:
                pid2 = pids[j]
                if sg.has_edge(pid1, pid2):
                    pid1_pid2_edge = sg[pid1][pid2]
                    pid1_pid2_edge['text'] = cos_similarity
                else:
                    sg.add_edge(pid1, pid2, text=cos_similarity)

    # Citations
    for u, v in citations_graph.edges:
        if sg.has_edge(u, v):
            sg[u][v]['citation'] = 1
        else:
            sg.add_edge(u, v, citation=1)

    # Ensure all the papers are in the similarity graph graph
    for pid in pids:
        if not sg.has_node(pid):
            sg.add_node(pid)

    return sg


def sparse_graph(graph, max_edges_to_nodes, key='weight'):
    e = 1.0
    gs = _local_sparse(graph, e, key)
    # Limit total number of edges to estimate walk probabilities
    while e > 0.1 and gs.number_of_edges() / gs.number_of_nodes() > max_edges_to_nodes:
        e -= 0.1
        gs = _local_sparse(graph, e, key)
    logger.debug(f'Sparse graph e={e} nodes={gs.number_of_nodes()} edges={gs.number_of_edges()}')
    return gs


def _local_sparse(graph, e, key='weight'):
    assert 0 <= e <= 1, f'Sparsity parameter {e} should be in 0..1'
    if e == 1:
        return graph
    result = nx.Graph()
    neighbours = {node: set(graph.neighbors(node)) for node in graph.nodes}
    priority_queues = {node: PriorityQueue(maxsize=ceil(pow(len(neighbours[node]), e)))
                       for node in graph.nodes}
    for (u, v, s) in graph.edges(data=key):
        qu = priority_queues[u]
        if qu.full():
            qu.get()  # Removes the element with lowest similarity
        qu.put((s, v))
        qv = priority_queues[v]
        if qv.full():
            qv.get()  # Removes the element with lowest similarity
        qv.put((s, u))
    for u, q in priority_queues.items():
        while not q.empty():
            s, v = q.get()
            if not result.has_edge(u, v):
                result.add_edge(u, v, **graph.edges[u, v])
    # Ensure all the nodes present
    for v in graph.nodes:
        if not result.has_node(v):
            result.add_node(v)
    return result


def to_weighted_graph(graph, weight_func, key='weight'):
    logger.debug('Creating weighted graph')
    g = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = weight_func(data)
        if np.isnan(w):
            raise Exception(f'Weight is NaN {w}')
        elif w < 0:
            raise Exception(f'Weight is < 0 {w}')
        elif w != 0:
            g.add_edge(u, v, **{key: w})
    # Ensure all the nodes present
    for v in graph.nodes:
        if not g.has_node(v):
            g.add_node(v)
    return g


def layout_similarity_graph(weighted_similarity_graph, topic_min_size, max_edges_to_nodes=50):
    """
    :return: node_ids, node2vec embeddings, xs, ys
    """
    if weighted_similarity_graph.number_of_nodes() <= topic_min_size:
        logger.debug('Preparing spring layout for similarity graph')
        pos = nx.spring_layout(weighted_similarity_graph, weight='weight')
        nodes = [a for a, _ in pos.items()]
        xs = [v[0] for _, v in pos.items()]
        ys = [v[1] for _, v in pos.items()]
        return nodes, np.zeros(shape=(len(nodes), 0), dtype=np.float), xs, ys
    else:
        logger.debug('Preparing node2vec + tsne layout for similarity graph')
        # Limit edges to nodes ratio in sparse similarity graph
        gs = sparse_graph(weighted_similarity_graph, max_edges_to_nodes=max_edges_to_nodes)
        node_ids, node_embeddings = node2vec(gs)
        logger.debug('Apply TSNE transformation on node embeddings')
        tsne = TSNE(n_components=2, random_state=42)
        node_embeddings_2d = tsne.fit_transform(node_embeddings)
        return node_ids, node_embeddings, node_embeddings_2d[:, 0], node_embeddings_2d[:, 1]
