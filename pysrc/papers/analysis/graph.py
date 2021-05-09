import logging
from queue import PriorityQueue

import networkx as nx
import pandas as pd
from math import floor

logger = logging.getLogger(__name__)


def build_citation_graph(cit_df):
    G = nx.DiGraph()
    for index, row in cit_df.iterrows():
        v, u = row['id_out'], row['id_in']
        G.add_edge(v, u)
    return G


def build_similarity_graph(
        df, texts_similarity, citations_graph, cocit_df, bibliographic_coupling_df
):
    """
    Relationship graph is build using citation and text based methods.

    See papers:
    Which type of citation analysis generates the most accurate taxonomy of
    scientific and technical knowledge? (https://arxiv.org/pdf/1511.05078.pdf)
    ...bibliographic coupling (BC) was the most accurate,  followed by co-citation (CC).
    Direct citation (DC) was a distant third among the three...

    Sugiyama, K., Kan, M.Y.:
    Exploiting potential citation papers in scholarly paper recommendation. In: JCDL (2013)
    """
    pids = list(df['id'])

    result = nx.Graph()
    # NOTE: we use nodes id as String to avoid problems str keys in jsonify
    # during graph visualization

    # Co-citations
    for el in cocit_df[['cited_1', 'cited_2', 'total']].values:
        start, end, cocitation = str(el[0]), str(el[1]), float(el[2])
        result.add_edge(start, end, cocitation=cocitation)

    # Bibliographic coupling
    if len(bibliographic_coupling_df) > 0:
        for el in bibliographic_coupling_df[['citing_1', 'citing_2', 'total']].values:
            start, end, bibcoupling = str(el[0]), str(el[1]), float(el[2])
            if result.has_edge(start, end):
                result[start][end]['bibcoupling'] = bibcoupling
            else:
                result.add_edge(start, end, bibcoupling=bibcoupling)

    # Text similarity
    if len(df) >= 2:
        for i, pid1 in enumerate(df['id']):
            similarity_queue = texts_similarity[i]
            while not similarity_queue.empty():
                similarity, j = similarity_queue.get()
                pid2 = pids[j]
                if result.has_edge(pid1, pid2):
                    pid1_pid2_edge = result[pid1][pid2]
                    pid1_pid2_edge['text'] = similarity
                else:
                    result.add_edge(pid1, pid2, text=similarity)

    # Citations
    for u, v in citations_graph.edges:
        if result.has_edge(u, v):
            result[u][v]['citation'] = 1
        else:
            result.add_edge(u, v, citation=1)

    # Ensure all the papers are in the similarity graph graph
    for pid in pids:
        if not result.has_node(pid):
            result.add_node(pid)

    return result


def build_structure_graph(df, similarity_graph, similarity_func, comp_sparsity, between_comp_sparsity):
    """
    Structure graph is sparse representation of similarity graph.
    It uses topics a structure.
    * For all the topics we show locally sparse subgraph.
    * Add top similarity connections between different topics to keep overall structure.
    :param df: main dataframe with information about papers
    :param similarity_graph: similarity graph
    :param similarity_func: function to compute aggregated similarity for each edge
    :param comp_sparsity: Coefficient to compute local sparse graph for topics
    :param between_comp_sparsity: Coefficient to compute local sparse connections between components
    :return:
    """
    result = nx.Graph()
    topics_edges = {}  # Number of edges in locally sparse graph for topic
    topics_mean_similarities = {}  # Average similarity per component

    logger.debug('Processing topics local sparse graphs')
    for c in set(df['comp']):
        comp_df = df.loc[df['comp'] == c]
        logger.debug(f'Processing component {c}')
        topic_sparse = local_sparse(similarity_graph.subgraph(comp_df['id']), e=comp_sparsity)
        connected_components = [cc for cc in nx.connected_components(topic_sparse)]
        logger.debug(f'Connected components {connected_components}')
        # Build a map node -> connected group
        connected_map = {}
        for ci, cc in enumerate(connected_components):
            for node in cc:
                connected_map[node] = ci

        logger.debug('Processing edges within sparse graph')
        topic_similarity_sum = 0
        topic_similarity_n = 0
        topic_n = None
        for (pu, pv, d) in topic_sparse.edges(data=True):
            result.add_edge(pu, pv, **d)
            if topic_n is None:
                topic_n = int(df.loc[df['id'] == pu]['comp'].values[0])
                topics_edges[topic_n] = len(topic_sparse.edges)
            topic_similarity_n += 1
            topic_similarity_sum += d['similarity']
        topics_mean_similarities[topic_n] = topic_similarity_sum / max(1, topic_similarity_n)

    logger.debug('Ensure all the papers are processed')
    for pid in df['id']:
        if not result.has_node(pid):
            result.add_node(pid)

    logger.debug('Add top similarity edges between topics')
    sources = [None] * len(similarity_graph.edges)
    targets = [None] * len(similarity_graph.edges)
    similarities = [0.0] * len(similarity_graph.edges)
    for i, (u, v, data) in enumerate(similarity_graph.edges(data=True)):
        sources[i] = u
        targets[i] = v
        similarities[i] = similarity_func(data)
    similarity_df = pd.DataFrame(data={'source': sources, 'target': targets, 'similarity': similarities})

    logger.debug('Assign each paper with corresponding component / topic')
    similarity_topics_df = similarity_df.merge(df[['id', 'comp']], how='left', left_on='source', right_on='id') \
        .merge(df[['id', 'comp']], how='left', left_on='target', right_on='id')

    inter_topics = {}
    for i, row in similarity_topics_df.iterrows():
        pid1, c1, pid2, c2, similarity = row['id_x'], row['comp_x'], row['id_y'], row['comp_y'], row['similarity']
        if c1 == c2:
            continue  # Ignore same group
        if c2 > c1:
            c2, c1 = c1, c2  # Swap
        # Ignore between components similarities less than average within groups
        if similarity < (topics_mean_similarities.get(c1, 0) +
                         topics_mean_similarities.get(c2, 0)) / 2:
            continue
        if (c1, c2) not in inter_topics:
            # Do not add edges between topics more than minimum * scale factor
            max_connections = \
                int(min(topics_edges[c1], topics_edges[c2]) * between_comp_sparsity)
            pq = inter_topics[(c1, c2)] = PriorityQueue(maxsize=max_connections)
        else:
            pq = inter_topics[(c1, c2)]
        if pq.full():
            pq.get()  # Removes the element with lowest similarity
        pq.put((similarity, pid1, pid2))
    for pq in inter_topics.values():
        while not pq.empty():
            _, pid1, pid2 = pq.get()
            # Add edge with full info
            result.add_edge(pid1, pid2, **similarity_graph.edges[pid1, pid2])
    return result


def local_sparse(graph, e):
    assert 0 <= e <= 1, f'Sparsity parameter {e} should be in 0..1'
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
