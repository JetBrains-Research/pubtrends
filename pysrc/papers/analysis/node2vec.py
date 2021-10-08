import logging

import numpy as np
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


def node2vec(graph, p=0.5, q=2.0, walk_length=32, walks_per_node=5, vector_size=32, key='weight', seed=42):
    """
    :param graph: Undirected weighted networkx graph
    :param p: Defines (unormalised) probability, 1/p, of returning to source node
    :param q: Defines (unormalised) probability, 1/q, for moving away from source node
    :param walk_length: Walk length for each node. Walk stops preliminary if no neighbors found
    :param walks_per_node: Number of walk actions performed starting from each node
    :param vector_size: Resulting embedding size
    :param key: key for weight
    :param seed: seed
    :returns indices and matrix of embeddings
    """
    logger.debug('Precomputing random walk probabilities')
    probabilities1, probabilities2 = _precompute(graph, key, p, q)

    logger.debug('Performing random walks')
    walks = _random_walks(
        graph, probabilities1, probabilities2,
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


def _precompute(graph, key='weight', p=0.5, q=2.0):
    """
    :param graph: Undirected weighted networkx graph
    :param key: weight key for edge
    :param p: Defines (unormalised) probability, 1/p, of returning to source node
    :param q: Defines (unormalised) probability, 1/q, for moving away from source node
    :return: probabilities on first step, and probabilities on steps 2+
    """
    probabilities1 = dict()  # No returning back option
    probabilities2 = {node: dict() for node in graph.nodes()}

    for i, node in enumerate(graph.nodes()):
        if i % 100 == 1:
            logger.debug(f'Analyzed probabilities for {i} nodes')
        neighbors = list(graph.neighbors(node))

        first_travel_weights = []
        for neighbor in neighbors:
            first_travel_weights.append(graph[node][neighbor].get(key, 1))
        probabilities1[node] = _normalize(first_travel_weights)

        for neighbor in neighbors:
            walk_weights = []
            for neighbor2 in graph.neighbors(neighbor):
                if neighbor2 == node:  # Backwards probability
                    ss_weight = graph[neighbor][neighbor2].get(key, 1) * 1 / p
                elif neighbor2 in graph[node]:  # If the neighbor is connected to the node
                    ss_weight = graph[neighbor][neighbor2].get(key, 1)
                else:
                    ss_weight = graph[neighbor][neighbor2].get(key, 1) * 1 / q
                walk_weights.append(ss_weight)
            probabilities2[neighbor][node] = _normalize(walk_weights)

    return probabilities1, probabilities2


def _normalize(values):
    """
    Normalize values, so that sum = 1
    """
    values = np.asarray(values).astype('float64')
    return values / values.sum()


def _random_walks(
        graph,
        probabilities1,
        probabilities2,
        walks_per_node,
        walk_length,
        seed=None
):
    np.random.seed(seed)
    walks = []
    for i in range(walks_per_node):
        logger.debug(f'Generating walk {i+1}')

        # Start a random walk from every node
        for node in graph.nodes():
            # Perform walk
            walk = [node]
            while len(walk) < walk_length:
                neighbors = list(graph.neighbors(walk[-1]))
                # Dead end nodes
                if not len(neighbors):
                    break
                if len(walk) == 1:
                    step_probabilities = probabilities1[walk[-1]]
                else:
                    step_probabilities = probabilities2[walk[-1]][walk[-2]]
                if len(neighbors) != len(step_probabilities):
                    raise Exception(f'Illegal probabilities for node {node}, '
                                    f'neighbors size {len(neighbors)}, '
                                    f'probabilities {len(step_probabilities)}')
                walk.append(np.random.choice(neighbors, size=1, p=step_probabilities)[0])
            walks.append(walk)
    return walks
