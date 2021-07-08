import logging
from queue import PriorityQueue

import networkx as nx
from math import floor
from node2vec import Node2Vec

# Avoid info message about compilation flags
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

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


def local_sparse(graph, e):
    assert 0 <= e <= 1, f'Sparsity parameter {e} should be in 0..1'
    if e == 1:
        return graph
    result = nx.Graph()
    neighbours = {node: set(graph.neighbors(node)) for node in graph.nodes}
    sim_queues = {node: PriorityQueue(maxsize=floor(pow(len(neighbours[node]), e)))
                  for node in graph.nodes}
    for (u, v, s) in graph.edges(data='similarity'):
        qu = sim_queues[u]
        if qu.full():
            qu.get()  # Removes the element with lowest similarity
        qu.put((s, v))
        qv = sim_queues[v]
        if qv.full():
            qv.get()  # Removes the element with lowest similarity
        qv.put((s, u))
    for u, q in sim_queues.items():
        while not q.empty():
            s, v = q.get()
            if not result.has_edge(u, v):
                result.add_edge(u, v, **graph.edges[u, v])
    return result


def node2vec(graph, weight_func, walk_length, walks_per_node, vector_size):
    logger.debug('Creating weighted graph')
    g_weighted = nx.Graph()
    for u, v, data in graph.edges(data=True):
        g_weighted.add_edge(u, v, weight=weight_func(data))

    # Ensure all the nodes present
    for v in graph.nodes:
        if not g_weighted.has_node(v):
            g_weighted.add_node(v)

    logger.debug('Performing random walks')
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    n2v = Node2Vec(
        g_weighted, p=0.5, q=2.0,
        dimensions=vector_size, walk_length=walk_length, num_walks=walks_per_node, workers=1,
        quiet=True
    )
    logger.debug('Performing word2vec emdeddings')
    model = n2v.fit(window=5, min_count=0, workers=1)
    return model.wv.index_to_key, model.wv.vectors