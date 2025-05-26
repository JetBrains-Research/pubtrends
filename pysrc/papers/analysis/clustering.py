import logging
from collections import Counter

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

def networkx_to_igraph(gnx, key):
    # Extract edges
    edges = list(gnx.edges())
    nodes = list(gnx.nodes())

    # Create a mapping from node labels to consecutive integers
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    inv_node_mapping = {v: k for k, v in node_mapping.items()}

    # Remap edges to integer IDs
    edges_igraph = [(node_mapping[u], node_mapping[v]) for u, v in edges]

    # Create igraph graph
    gig = ig.Graph(edges=edges_igraph, directed=False)

    # Optional: Add key:value if present
    if nx.get_edge_attributes(gnx, key):
        values = [gnx[u][v].get(key, 1.0) for u, v in edges]
        gig.es[key] = values
    else:
        gig.es[key] = [1.0] * len(edges)
    return gig, node_mapping, inv_node_mapping

def cluster_papers_graph(pids, g, min_cluster_size=10):
    """
    :param g: graph
    :param min_cluster_size, %
    :return: List[cluster]
    """
    min_size = int(len(g.nodes()) * min_cluster_size * 0.01)
    gig, node_map, inv_map = networkx_to_igraph(g, 'similarity')
    # 0.2 - Few, larger clusters
    # 0.5 - Moderate number of clusters
    # 1.0 - Many small, fine-grained clusters
    # 1.5 - Over clustering

    # Compute resolution that produces minimal average cluster size
    resolution = 0.1
    part = leidenalg.find_partition(gig, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
    while resolution <= 2 and np.mean([cnt for _, cnt in Counter(part.membership).most_common()]) >= min_size:
        resolution += 0.1
        part = leidenalg.find_partition(
            gig, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, weights='similarity'
        )
    part = leidenalg.find_partition(
        gig, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution - 0.1, weights='similarity'
    )

    clusters = [0] * len(pids)
    pid_to_idx = {pid : i for i, pid in enumerate(pids)}
    for i, c in enumerate(part.membership):
        clusters[pid_to_idx[inv_map[i]]] = c
    return reorder_by_size(clusters)


def reorder_by_size(clusters):
    clusters_counter = Counter(clusters)
    logger.debug('Reorder clusters by size descending')
    min_size = clusters_counter.most_common()[-1][1]
    logger.debug(f'Min cluster size = {min_size}')
    reorder_map = {c: i for i, (c, _) in enumerate(clusters_counter.most_common())}
    result = [reorder_map[c] for c in clusters]
    return result

