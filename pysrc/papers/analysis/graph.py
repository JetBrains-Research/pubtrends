import logging
from math import ceil
from queue import PriorityQueue

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def build_citation_graph(cit_df):
    cg = nx.DiGraph()
    for index, row in cit_df.iterrows():
        v, u = row['id_out'], row['id_in']
        cg.add_edge(v, u)
    return cg


def build_similarity_graph(
        df, texts_similarity, citations_graph, cocit_df, bibliographic_coupling_df
):
    """
    Similarity graph is built using citation and text based methods.

    See papers:
    Which type of citation analysis generates the most accurate taxonomy of
    scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
    ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
    Direct citation (DC) was a distant third among the three...

    Sugiyama, K., Kan, M.Y.:
    Exploiting potential citation papers in scholarly paper recommendation. In: JCDL (2013)
    """
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
            similarity_queue = texts_similarity[i]
            while not similarity_queue.empty():
                similarity, j = similarity_queue.get()
                pid2 = pids[j]
                if sg.has_edge(pid1, pid2):
                    pid1_pid2_edge = sg[pid1][pid2]
                    pid1_pid2_edge['text'] = similarity
                else:
                    sg.add_edge(pid1, pid2, text=similarity)

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


def local_sparse(graph, e, key='weight'):
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
