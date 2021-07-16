import logging

import networkx as nx
import numpy as np

# Avoid info message about compilation flags
import tensorflow as tf
from gensim.models import Word2Vec

tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


def _precompute(graph, weight_key='weight', p=0.5, q=2.0):
    """
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    """
    adjacency_map = dict()
    probabilities_first_step = dict()  # No returning back option
    probabilities = {node: dict() for node in graph.nodes()}

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        adjacency_map[node] = neighbors

        first_travel_weights = []
        for neighbor in neighbors:
            first_travel_weights.append(graph[node][neighbor].get(weight_key, 1))
        probabilities_first_step[node] = _normalize(first_travel_weights)

        for neighbor in neighbors:
            walk_weights = []
            for neighbor2 in graph.neighbors(neighbor):
                if neighbor2 == node:  # Backwards probability
                    ss_weight = graph[neighbor][neighbor2].get(weight_key, 1) * 1 / p
                elif neighbor2 in graph[node]:  # If the neighbor is connected to the node
                    ss_weight = graph[neighbor][neighbor2].get(weight_key, 1)
                else:
                    ss_weight = graph[neighbor][neighbor2].get(weight_key, 1) * 1 / q
                walk_weights.append(ss_weight)
            probabilities[neighbor][node] = _normalize(walk_weights)

    return adjacency_map, probabilities_first_step, probabilities


def _normalize(values):
    values = np.array(values)
    if values.sum() != 0:
        return values / values.sum()
    else:
        return values


def _random_walks(nodes, adjacency_map, probabilities_first_step, probabilities, walks_per_node, walk_length, seed=42):
    np.random.seed(seed)
    walks = []
    for i in range(walks_per_node):
        # Start a random walk from every node
        for node in nodes:
            # Perform walk
            walk = [node]
            while len(walk) < walk_length:
                neighbors = adjacency_map[walk[-1]]
                # Dead end nodes
                if not len(neighbors):
                    break
                next_probabilities = probabilities_first_step[walk[-1]] \
                    if len(walk) == 1 else probabilities[walk[-1]][walk[-2]]
                walk.append(np.random.choice(neighbors, size=1, p=next_probabilities)[0])
            walks.append(walk)
    return walks


def node2vec(graph, weight_func, walk_length=100, walks_per_node=10, vector_size=64, seed=42):
    logger.debug('Creating weighted graph')
    g_weighted = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = weight_func(data)
        if np.isnan(w):
            raise Exception(f"ERROR weight {w}")
        g_weighted.add_edge(u, v, weight=w)

    # Ensure all the nodes present
    for v in graph.nodes:
        if not g_weighted.has_node(v):
            logger.info(f'Adding isolated vertex {v}')
            g_weighted.add_node(v)

    logger.debug('Precomputing random walk probabilities')
    adjacency_map, probabilities_first_step, probabilities = _precompute(g_weighted)
    walks = _random_walks(
        list(g_weighted.nodes), adjacency_map, probabilities_first_step, probabilities,
        walks_per_node, walk_length,
        seed=seed
    )

    logger.debug('Performing word2vec embeddings')
    logging.getLogger('node2vec.py').setLevel('ERROR')  # Disable logging
    w2v = Word2Vec(
        walks, vector_size=vector_size, window=5, min_count=0, sg=1, workers=1, epochs=1, seed=seed
    )
    # Retrieve node embeddings and corresponding subjects
    return w2v.wv.index_to_key, w2v.wv.vectors
