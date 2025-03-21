import logging
import numpy as np
from gensim.models import Word2Vec

from pysrc.config import EMBEDDINGS_VECTOR_LENGTH, NODE2VEC_P, NODE2VEC_Q, NODE2VEC_WALK_LENGTH, \
    NODE2VEC_WALKS_PER_NODE, NODE2VEC_WORD2VEC_WINDOW, NODE2VEC_WORD2VEC_EPOCHS

logger = logging.getLogger(__name__)

def node2vec(
        ids, graph, p=NODE2VEC_P, q=NODE2VEC_Q,
        walk_length=NODE2VEC_WALK_LENGTH, walks_per_node=NODE2VEC_WALKS_PER_NODE,
        vector_size=EMBEDDINGS_VECTOR_LENGTH, key='weight', seed=42
):
    """
    :param ids: Ids or nodes for embedding
    :param graph: Undirected weighted networkx graph
    :param p: Defines probability, 1/p, of returning to source node
    :param q: Defines probability, 1/q, for moving away from source node
    :param walk_length: Walk length for each node. Walk stops preliminary if no neighbors found
    :param walks_per_node: Number of walk actions performed starting from each node
    :param vector_size: Resulting embedding size
    :param key: key for weight
    :param seed: seed
    :returns matrix of embeddings
    """
    logger.debug('Precomputing random walk probabilities')
    probabilities_first_step, probabilities_next_step = _precompute(graph, key, p, q)

    logger.debug('Performing random walks')
    walks = _random_walks(
        graph, probabilities_first_step, probabilities_next_step,
        walks_per_node, walk_length,
        seed=seed
    )
    logger.debug('Performing word2vec embeddings')
    logging.getLogger('node2vec.py').setLevel('ERROR')  # Disable logging
    w2v = Word2Vec(
        walks, vector_size=vector_size,
        window=NODE2VEC_WORD2VEC_WINDOW,
        min_count=0, sg=1, workers=1, epochs=NODE2VEC_WORD2VEC_EPOCHS, seed=seed
    )
    logger.debug('Retrieve word embeddings, corresponding subjects and reorder according to ids')
    node_ids, node_embeddings = w2v.wv.index_to_key, w2v.wv.vectors
    indx = {pid: i for i, pid in enumerate(node_ids)}
    embeddings = np.array([
        node_embeddings[indx[pid]] if pid in indx else np.zeros(node_embeddings.shape[1])  # Process missing
        for pid in ids
    ])
    logger.debug('put 0 when edge has no neighbors')
    for i, node in enumerate(ids):
        if len(list(graph.neighbors(node))) == 0:
            embeddings[i] = np.zeros(vector_size)
    return embeddings


def _precompute(graph, key, p, q):
    """
    :param graph: Undirected weighted networkx graph
    :param key: weight key for edge
    :param p: Defines probability, 1/p, of returning to source node
    :param q: Defines probability, 1/q, for moving away from source node
    :return: probabilities on first step, and probabilities on steps 2+
    """
    probabilities_first_step = dict()  # No returning back option
    probabilities_next_step = {node: dict() for node in graph.nodes()}

    for i, node in enumerate(graph.nodes()):
        if i % 100 == 1:
            logger.debug(f'Analyzed probabilities for {i} nodes')
        first_step_weights = [graph[node][neighbor].get(key) for neighbor in (graph.neighbors(node))]
        probabilities_first_step[node] = _normalize(first_step_weights)

        for neighbor in graph.neighbors(node):
            walk_weights = [_next_step_weight(graph, key, node, neighbor, neighbor2, p, q)
                            for neighbor2 in graph.neighbors(neighbor)]
            probabilities_next_step[neighbor][node] = _normalize(walk_weights)

    return probabilities_first_step, probabilities_next_step


def _next_step_weight(graph, key, node, neighbor, neighbor2, p, q):
    if neighbor2 == node:  # Backwards probability
        return graph[neighbor][neighbor2].get(key) * 1 / p
    elif neighbor2 in graph[node]:  # If the neighbor is connected to the node
        return graph[neighbor][neighbor2].get(key)
    else:
        return graph[neighbor][neighbor2].get(key) * 1 / q


def _normalize(values):
    """ Normalize values, so that sum = 1 """
    values = np.asarray(values).astype('float64')
    return values / values.sum()


def _random_walks(
        graph,
        probabilities_first_step,
        probabilities_next_step,
        walks_per_node,
        walk_length,
        seed=None
):
    """ Perform random walks with given probabilities """
    np.random.seed(seed)
    walks = []
    for i in range(walks_per_node):
        logger.debug(f'Generating walk {i + 1}')

        # Start a random walk from every node
        for node in graph.nodes():
            # Perform walk
            walk = [node]
            while len(walk) < walk_length:
                neighbors = list(graph.neighbors(walk[-1]))
                # Dead end nodes
                if len(neighbors) == 0:
                    break
                if len(walk) == 1:
                    step_probabilities = probabilities_first_step[walk[-1]]
                else:
                    step_probabilities = probabilities_next_step[walk[-1]][walk[-2]]
                if len(neighbors) != len(step_probabilities):
                    raise Exception(f'Illegal probabilities for node {node}, '
                                    f'neighbors size {len(neighbors)}, '
                                    f'probabilities {len(step_probabilities)}')
                walk.append(np.random.choice(neighbors, size=1, p=step_probabilities)[0])
            walks.append(walk)
    return walks
