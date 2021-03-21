import numpy as np

from sklearn.metrics.cluster import contingency_matrix, v_measure_score


def pd_score(labels_true, labels_pred):
    """
    For each ground truth class:
    PD = Purity * Diversity
    Purity = sum(Normalized Size * Homogeneity) for all predicted clusters
    Diversity is 1 when the class was not split into multiple clusters, otherwise less, min 0
    """
    contingency = contingency_matrix(labels_true, labels_pred)

    row_sums = contingency.sum(axis=1)
    contingency_row = contingency / row_sums[:, np.newaxis]
    col_sums = contingency.sum(axis=0)
    contingency_col = contingency / col_sums[np.newaxis, :]

    diversity = np.sqrt(np.diag(contingency_row @ contingency_row.T))
    purity = np.diag(contingency_row @ contingency_col.T)
    size_weights = row_sums / sum(row_sums)
    return np.sum(purity * diversity * size_weights)


def reg_v_score(labels_true, labels_pred, reg=0.01):
    v_score = v_measure_score(labels_true, labels_pred)
    n_clusters = len(set(labels_pred))
    return v_score - reg * n_clusters
