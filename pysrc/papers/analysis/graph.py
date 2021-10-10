import logging
from math import ceil
from queue import PriorityQueue

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def build_papers_graph(df, cit_df, cocit_df, bibliographic_coupling_df):
    pids = list(df['id'])

    sg = nx.Graph()
    # NOTE: we use nodes id as String to avoid problems str keys in jsonify
    # during graph visualization

    # Co-citations
    for start, end, cocitation in zip(cocit_df['cited_1'], cocit_df['cited_2'], cocit_df['total']):
        sg.add_edge(start, end, cocitation=cocitation)

    # Bibliographic coupling
    if len(bibliographic_coupling_df) > 0:
        for start, end, bibcoupling in zip(bibliographic_coupling_df['citing_1'],
                                           bibliographic_coupling_df['citing_2'],
                                           bibliographic_coupling_df['total']):
            if sg.has_edge(start, end):
                sg[start][end]['bibcoupling'] = bibcoupling
            else:
                sg.add_edge(start, end, bibcoupling=bibcoupling)

    # Citations
    for start, end in zip(cit_df['id_out'], cit_df['id_in']):
        if sg.has_edge(start, end):
            sg[start][end]['citation'] = 1
        else:
            sg.add_edge(start, end, citation=1)

    # Ensure all the papers are in the graph
    for pid in pids:
        if not sg.has_node(pid):
            sg.add_node(pid)

    return sg


def sparse_graph(graph, max_edges_to_nodes=30, key='weight'):
    logger.debug(f'Limit total number of edges to max_edges_to_nodes={max_edges_to_nodes}')
    e = 1.0
    gs = _local_sparse(graph, e, key)
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
