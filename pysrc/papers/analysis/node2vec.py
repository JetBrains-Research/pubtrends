import logging

import numpy as np
# Avoid info message about compilation flags
import tensorflow as tf
from gensim.models import Word2Vec

tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


def node2vec(graph, p=0.5, q=2.0, walk_length=100, walks_per_node=10, vector_size=64, seed=42):
    """
    :param graph: Undirected networkx graph, 'weight' value of each edge indicates similarity
    :param p: Defines (unormalised) probability, 1/p, of returning to source node
    :param q: Defines (unormalised) probability, 1/q, for moving away from source node
    :param walk_length: Walk length for each node. Walk stops preliminary if no neighbors found
    :param walks_per_node: Number of walk actions performed starting from each node
    :param vector_size: Resulting embedding size
    :param seed: seed
    :returns indices and matrix of embeddings
    """

    logger.debug('Precomputing random walk probabilities')
    adjacency_map, probabilities_first_step, probabilities = _precompute(graph, p=p, q=q)
    walks = _random_walks(
        list(graph.nodes), adjacency_map, probabilities_first_step, probabilities,
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


def _precompute(graph, p=0.5, q=2.0):
    """
    :param graph: Undirected networkx graph, 'weight' value of each edge indicates similarity
    :param p: Defines (unormalised) probability, 1/p, of returning to source node
    :param q: Defines (unormalised) probability, 1/q, for moving away from source node
    :return: adjacency map, probabilities on first step, and probabilities on 2+ steps
    """
    adjacency_map = dict()
    probabilities_first_step = dict()  # No returning back option
    probabilities = {node: dict() for node in graph.nodes()}

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        adjacency_map[node] = neighbors

        first_travel_weights = []
        for neighbor in neighbors:
            first_travel_weights.append(graph[node][neighbor].get('weight', 1))
        probabilities_first_step[node] = _normalize(first_travel_weights)

        for neighbor in neighbors:
            walk_weights = []
            for neighbor2 in graph.neighbors(neighbor):
                if neighbor2 == node:  # Backwards probability
                    ss_weight = graph[neighbor][neighbor2].get('weight', 1) * 1 / p
                elif neighbor2 in graph[node]:  # If the neighbor is connected to the node
                    ss_weight = graph[neighbor][neighbor2].get('weight', 1)
                else:
                    ss_weight = graph[neighbor][neighbor2].get('weight', 1) * 1 / q
                walk_weights.append(ss_weight)
            probabilities[neighbor][node] = _normalize(walk_weights)

    return adjacency_map, probabilities_first_step, probabilities


def _normalize(values):
    """
    Normalize values, so that sum is guaranteed < 1
    """
    values = np.asarray(values).astype('float64')
    s = values.sum()
    if s == 1:
        return values
    else:
        return values / (s + 1e-10)  # Ensure that sum is <= 1


def _random_walks(
        nodes,
        adjacency_map,
        probabilities_first_step,
        probabilities,
        walks_per_node,
        walk_length,
        seed=42
):
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
                if len(neighbors) != len(next_probabilities):
                    raise Exception(f'Illegal probabilities for node {node}, '
                                    f'neighbors size {len(neighbors)}, '
                                    f'probabilities {len(next_probabilities)}')
                # Workaround for np.random.choice issue: probabilities do not sum to 1
                # Use np.random.multinomial with constraint that sum to <=1
                choice = np.random.multinomial(1, next_probabilities)[0]
                if choice < len(neighbors):
                    walk.append(neighbors[choice])
                else:
                    # Fallback for last artificial choice in multinomial
                    walk.append(np.random.choice(neighbors))
            walks.append(walk)
    return walks
