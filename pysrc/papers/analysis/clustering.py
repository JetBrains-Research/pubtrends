import logging
from collections import Counter

from sklearn.cluster import KMeans, AgglomerativeClustering

from pysrc.config import MAX_AGGLOMERATIVE_CLUSTERING

logger = logging.getLogger(__name__)


def cluster_and_sort(x, n_clusters):
    """
    :param x: object representations (X x Features)
    :param n_clusters:
    :return: List[cluster], Hierarchical dendrogram of splits?
    """
    logger.debug(f'Looking for clusters={n_clusters}')
    if x.shape[0] <= n_clusters or x.shape[1] == 0:
        return [0] * x.shape[0], None

    if x.shape[0] <= MAX_AGGLOMERATIVE_CLUSTERING:
        model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=None).fit(x)
        labels = model.labels_
        hierarchy = model.children_
    else:
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(x)
        hierarchy = None

    clusters_counter = Counter(labels)
    min_cluster = clusters_counter.most_common()[-1][1]
    logger.debug(f'Clusters = {n_clusters}, min cluster size = {min_cluster}')
    clusters = reorder_by_size(labels)
    logger.debug(f'Clusters sizes: {Counter(clusters)}')
    return clusters, hierarchy


def reorder_by_size(clusters):
    clusters_counter = Counter(clusters)
    logger.debug('Reorder clusters by size descending')
    reorder_map = {c: i for i, (c, _) in enumerate(clusters_counter.most_common())}
    result = [reorder_map[c] for c in clusters]
    return result
