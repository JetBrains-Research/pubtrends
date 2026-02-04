import logging
from collections import Counter

from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def cluster_and_sort(x, n_clusters):
    """
    :param x: object representations (X x Features)
    :param n_clusters:
    :return: List[cluster], Hierarchical dendrogram of splits.
    """
    logger.debug(f'Looking for clusters={n_clusters}')
    if x.shape[0] <= n_clusters or x.shape[1] == 0:
        return [0] * x.shape[0], None

    model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=None).fit(x)
    clusters_counter = Counter(model.labels_)
    min_cluster = clusters_counter.most_common()[-1][1]
    logger.debug(f'Clusters = {n_clusters}, min cluster size = {min_cluster}')

    # Estimate hierarchical clustering possibilities
    # Closer to 1 → Better hierarchical clustering.
    z = linkage(x, metric="cosine", method="average")
    coph_corr, _ = cophenet(z, pdist(x))
    print(f'Cophenetic Correlation Coefficient: {coph_corr}, the closer to 1, the better hierarchical clustering')

    clusters = reorder_by_size(model.labels_)
    logger.debug(f'Clusters sizes: {Counter(clusters)}')
    return clusters, model.children_


def reorder_by_size(clusters):
    clusters_counter = Counter(clusters)
    logger.debug('Reorder clusters by size descending')
    reorder_map = {c: i for i, (c, _) in enumerate(clusters_counter.most_common())}
    result = [reorder_map[c] for c in clusters]
    return result
