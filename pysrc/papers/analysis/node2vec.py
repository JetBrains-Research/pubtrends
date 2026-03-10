import concurrent.futures
import logging
import multiprocessing
from threading import Lock

import numpy as np
from gensim.models import Word2Vec

from pysrc.config import NODE2VEC_P, NODE2VEC_Q, NODE2VEC_WALK_LENGTH, \
    NODE2VEC_WALKS_PER_NODE, NODE2VEC_WORD2VEC_WINDOW, NODE2VEC_WORD2VEC_EPOCHS, NODE2VEC_EMBEDDINGS_VECTOR_LENGTH, \
    ANALYSIS_CHUNK

logger = logging.getLogger(__name__)

# Lock for thread-safe random number generation
_RNG_LOCK = Lock()

def node2vec(
        ids, graph, key, p=NODE2VEC_P, q=NODE2VEC_Q,
        walk_length=NODE2VEC_WALK_LENGTH, walks_per_node=NODE2VEC_WALKS_PER_NODE,
        vector_size=NODE2VEC_EMBEDDINGS_VECTOR_LENGTH, seed=42
):
    """
    :param ids: Ids or nodes for embedding
    :param graph: Undirected weighted networkx graph
    :param key: key for weight
    :param p: Defines probability, 1/p, of returning to source node
    :param q: Defines probability, 1/q, for moving away from source node
    :param walk_length: Walk length for each node. Walk stops preliminary if no neighbors found
    :param walks_per_node: Number of walk actions performed starting from each node
    :param vector_size: Resulting embedding size
    :param seed: seed
    :returns matrix of embeddings
    """
    logger.debug('Precomputing random walk probabilities')
    probabilities_first_step, probabilities_next_step, neighbors_cache = _precompute(graph, key, p, q)

    logger.debug('Performing random walks')
    walks = _random_walks(
        neighbors_cache, probabilities_first_step, probabilities_next_step,
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
    nas = 0
    for i, node in enumerate(ids):
        if len(list(graph.neighbors(node))) == 0:
            nas += 1
            embeddings[i] = np.ones(vector_size)
    logger.debug(f'Total {nas} nodes without neighbors')
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

    # Cache neighbors and node connectivity for faster lookup
    neighbors_cache = {node: list(graph.neighbors(node)) for node in graph.nodes()}
    node_set = {node: set(neighbors_cache[node]) for node in graph.nodes()}

    for i, node in enumerate(graph.nodes()):
        if i % ANALYSIS_CHUNK == 0:
            logger.debug(f'Analyzed probabilities for {i + 1} nodes')

        node_neighbors = neighbors_cache[node]
        if not node_neighbors:
            probabilities_first_step[node] = np.array([])
            continue

        # Vectorized first step weights
        first_step_weights = np.array([graph[node][neighbor].get(key) for neighbor in node_neighbors])
        probabilities_first_step[node] = _normalize(first_step_weights)

        for neighbor in node_neighbors:
            neighbor2_list = neighbors_cache[neighbor]
            if not neighbor2_list:
                probabilities_next_step[neighbor][node] = np.array([])
                continue

            # Vectorized next step weights calculation
            walk_weights = np.array([graph[neighbor][neighbor2].get(key) for neighbor2 in neighbor2_list])

            # Apply p and q factors
            for idx, neighbor2 in enumerate(neighbor2_list):
                if neighbor2 == node:  # Backwards probability
                    walk_weights[idx] *= 1 / p
                elif neighbor2 not in node_set[node]:  # Moving away
                    walk_weights[idx] *= 1 / q
                # else: neighbor is connected to node, weight unchanged

            probabilities_next_step[neighbor][node] = _normalize(walk_weights)

    return probabilities_first_step, probabilities_next_step, neighbors_cache


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
        neighbors_cache,
        probabilities_first_step,
        probabilities_next_step,
        walks_per_node,
        walk_length,
        seed=None
):
    """ Perform random walks with given probabilities using parallel processing """
    nodes = list(neighbors_cache.keys())
    max_workers = multiprocessing.cpu_count()

    logger.debug(f'Performing random walks with {max_workers} processes')

    # Prepare work for parallel processing - create tasks per walk iteration
    tasks = []
    for walk_idx in range(walks_per_node):
        # Each task gets a unique seed for reproducibility
        task_seed = (seed + walk_idx * 10000) if seed is not None else None
        tasks.append((
            nodes,
            neighbors_cache,
            probabilities_first_step,
            probabilities_next_step,
            walk_length,
            task_seed
        ))

    # Execute walks in parallel using processes
    all_walks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_perform_walks_for_all_nodes, tasks)
        for batch_walks in results:
            all_walks.extend(batch_walks)

    logger.debug(f'Generated {len(all_walks)} walks')
    return all_walks


def _perform_walks_for_all_nodes(task):
    """Perform one random walk from each node (used in parallel multiprocessing)"""
    nodes, neighbors_cache, probabilities_first_step, probabilities_next_step, walk_length, seed = task

    # Set seed for this process
    if seed is not None:
        np.random.seed(seed)

    walks = []
    for node in nodes:
        # Perform walk
        walk = [node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = neighbors_cache[current_node]

            # Dead end nodes
            if len(neighbors) == 0:
                break

            if len(walk) == 1:
                step_probabilities = probabilities_first_step[current_node]
            else:
                step_probabilities = probabilities_next_step[current_node][walk[-2]]

            if len(neighbors) != len(step_probabilities):
                raise Exception(f'Illegal probabilities for node {node}, '
                                f'neighbors size {len(neighbors)}, '
                                f'probabilities {len(step_probabilities)}')

            # Random choice using numpy
            next_node = np.random.choice(neighbors, p=step_probabilities)
            walk.append(int(next_node) if isinstance(next_node, np.integer) else next_node)

        walks.append(walk)
    return walks
