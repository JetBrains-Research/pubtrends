from celery import Celery

celery_app = Celery('grid_search', backend='redis://localhost:6379', broker='redis://localhost:6379')

# Save all generated partitions for further investigation
# (might consume a LOT of space)
SAVE_PARTITION = True

weight_range = [1, 2, 4, 8]
topic_min_size = [5, 10]
topics_max_number = [10, 20]
resolution = [0.6, 0.8, 1]

param_grid = [
    {
        'method': ['node2vec'],
        'similarity_bibliographic_coupling': weight_range.copy(),
        'similarity_cocitation': [1],
        'similarity_citation': weight_range.copy(),
        'similarity_text': weight_range.copy(),
        'walk_length': [64],
        'walks_per_node': [32],
        'vector_size': [32],
        'topic_min_size': topic_min_size.copy(),
        'topics_max_number': topics_max_number.copy(),
        'check_cached_data': [False],  # for local testing, cache should work fine
    },
    {
        'similarity_bibliographic_coupling': weight_range.copy(),
        'similarity_cocitation': [1],
        'similarity_citation': weight_range.copy(),
        'similarity_text': weight_range.copy(),
        'method': ['louvain'],
        'resolution': resolution.copy(),
    },
    {
        'similarity_bibliographic_coupling': [1],
        'similarity_cocitation': [0],
        'similarity_citation': weight_range.copy(),
        'similarity_text': weight_range.copy(),
        'method': ['louvain'],
        'resolution': resolution.copy(),
    },
    {
        'similarity_bibliographic_coupling': [0],
        'similarity_cocitation': [0],
        'similarity_citation': [1],
        'similarity_text': weight_range.copy(),
        'method': ['louvain'],
        'resolution': resolution.copy(),
    },
    {
        'method': ['lda'],
        'max_df': [0.8],
        'min_df': [0.001],
        'max_features': [1000],
        'topic_min_size': topic_min_size.copy(),
        'topics_max_number': topics_max_number.copy(),
        'check_cached_data': [False],  # for local testing, cache should work fine
    }
]

from sklearn.model_selection import ParameterGrid

print('Parameters grid size', len(ParameterGrid(param_grid)))

import logging

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.model_selection import ParameterGrid

from utils.analysis import get_direct_references_subgraph, align_clusterings_for_sklearn
from utils.io import load_analyzer, load_clustering, get_review_pmids
from utils.metrics import pd_score, reg_v_score
from utils.preprocessing import preprocess_clustering, get_clustering_level

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# LDA
from collections import Counter
from functools import lru_cache

from pysrc.papers.analysis.text import build_corpus

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def cluster_lda(X, min_cluster_size, max_clusters):
    logging.debug('Looking for an appropriate number of clusters,'
                  f'min_cluster_size={min_cluster_size}, max_clusters={max_clusters}')
    r = min(max_clusters, X.shape[0]) + 1
    l = 1

    if l >= r - 2:
        return [0] * X.shape[0]

    while l < r - 2:
        n_clusters = int((l + r) / 2)
        lda = LatentDirichletAllocation(n_components=n_clusters, random_state=SEED).fit(X)
        clusters = lda.transform(X).argmax(axis=1)
        clusters_counter = Counter(clusters)
        min_size = clusters_counter.most_common()[-1][1]
        num_clusters = len(clusters_counter.keys())
        if min_size < min_cluster_size or num_clusters > max_clusters:
            r = n_clusters + 1
        else:
            l = n_clusters

    logging.debug(f'Number of clusters = {n_clusters}')
    logging.debug('Reorder clusters by size descending')
    reorder_map = {c: i for i, (c, _) in enumerate(clusters_counter.most_common())}
    return [reorder_map[c] for c in clusters]


@lru_cache(maxsize=1000)
def preproc_lda(analyzer, subgraph, max_df, min_df, max_features):
    # Use only papers present in the subgraph
    subgraph_df = analyzer.df[analyzer.df.id.isin(subgraph.nodes)]
    node_ids = list(subgraph_df.id.values)

    # Build and vectorize corpus for LDA
    corpus = build_corpus(subgraph_df)
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, node_ids


def topic_analysis_lda(analyzer, subgraph, **settings):
    X, node_ids = preproc_lda(analyzer, subgraph,
                              max_df=settings['max_df'],
                              min_df=settings['min_df'],
                              max_features=settings['max_features'])

    # Check the cached data
    if settings['check_cached_data']:
        n = subgraph.number_of_nodes()
        assert X.shape[0] == n
        assert X.shape[1] <= settings['max_features']
        assert list(sorted(subgraph.nodes())) == sorted(node_ids)

    clusters = cluster_lda(X, min_cluster_size=settings['topic_min_size'],
                           max_clusters=settings['topics_max_number'])

    return dict(zip(node_ids, clusters))


# Louvain
import community

from pysrc.papers.utils import SEED

from nature_reviews.src.utils.analysis import get_similarity_func


def topic_analysis_louvain(analyzer, similarity_graph, **settings):
    """
    Performs clustering of similarity topics with different cluster resolution
    :param analyzer: is not used, only for consistency with other methods
    :param similarity_graph: Similarity graph
    :param settings: contains all tunable parameters
    :return: merged_partition
    """
    logging.debug('Compute aggregated similarity')
    similarity_func = get_similarity_func(settings['similarity_bibliographic_coupling'],
                                          settings['similarity_cocitation'],
                                          settings['similarity_citation'],
                                          settings['similarity_text'])
    for _, _, d in similarity_graph.edges(data=True):
        d['similarity'] = similarity_func(d)

    logging.debug('Graph clustering via Louvain community algorithm')
    partition_louvain = community.best_partition(
        similarity_graph, weight='similarity', random_state=SEED, resolution=settings['resolution']
    )

    return partition_louvain


# Node2vec
from pysrc.papers.analysis.graph import node2vec
from pysrc.papers.analysis.topics import cluster_embeddings


@lru_cache(maxsize=1000)
def pre_proc_node2vec(
        subgraph,
        similarity_bibliographic_coupling,
        similarity_cocitation,
        similarity_citation,
        similarity_text_citation,
        walk_length,
        walks_per_node,
        vector_size):
    similarity_func = get_similarity_func(similarity_bibliographic_coupling,
                                          similarity_cocitation,
                                          similarity_citation,
                                          similarity_text_citation)
    node_ids, node_embeddings = node2vec(subgraph,
                                         weight_func=similarity_func,
                                         walk_length=walk_length,
                                         walks_per_node=walks_per_node,
                                         vector_size=vector_size)
    return node_embeddings, node_ids


def topic_analysis_node2vec(analyzer, subgraph, **settings):
    """
    Rerun topic analysis based on node2vec for a given similarity graph and settings.
    """
    node_embeddings, node_ids = pre_proc_node2vec(
        subgraph, settings['similarity_bibliographic_coupling'], settings['similarity_cocitation'],
        settings['similarity_citation'], settings['similarity_text'], settings['walk_length'],
        settings['walks_per_node'], settings['vector_size'])

    if settings['check_cached_data']:
        n = subgraph.number_of_nodes()
        assert node_embeddings.shape[0] == n
        assert node_embeddings.shape[1] <= settings['vector_size']
        assert list(sorted(subgraph.nodes())) == sorted(node_ids)

    clusters, _ = cluster_embeddings(
        node_embeddings, settings['topic_min_size'], settings['topics_max_number']
    )
    return dict(zip(node_ids, clusters))


# Topic analysis
def topic_analysis(analyzer, subgraph, method, **method_params):
    """
    Returns partition - dictionary {pmid (str): cluster (int)}
    """
    if method == 'node2vec':
        return topic_analysis_node2vec(analyzer, subgraph, **method_params)
    elif method == 'lda':
        return topic_analysis_lda(analyzer, subgraph, **method_params)
    elif method == 'louvain':
        return topic_analysis_louvain(analyzer, subgraph, **method_params)
    else:
        raise ValueError('Unknown clustering method')


def run_grid_search(analyzer, subgraph, ground_truth, metrics, param_grid, save_partition=False):
    # Accumulate grid results for all hierarchy levels
    grid_results = []
    partitions = []

    parameter_grid = ParameterGrid(param_grid)
    grid_size = len(parameter_grid)
    for i, param_values in enumerate(parameter_grid):
        partition = topic_analysis(analyzer, subgraph, **param_values)

        if save_partition:
            param_partition = param_values.copy()
            param_partition['partition'] = partition
            partitions.append(param_partition)

        # Iterate over hierarchy levels to avoid re-calculating same clustering for different ground truth
        for level, ground_truth_partition in ground_truth.items():
            result = param_values.copy()
            result['level'] = level
            labels_true, labels_pred = align_clusterings_for_sklearn(partition, ground_truth_partition)
            result['n_clusters'] = len(set(labels_pred))

            # Evaluate different metrics
            for metric in metrics:
                result[metric.__name__] = metric(labels_true, labels_pred)

            grid_results.append(result)

        if (i + 1) % 10 == 0:
            print(f' {i + 1} / {grid_size}\n')
    print('\n')

    return grid_results, partitions


@celery_app.task(name='run_single_parameter')
def run_single_parameter(pmid):
    clustering = load_clustering(pmid)
    analyzer = load_analyzer(pmid)

    # Pre-calculate all hierarchy levels before grid search to avoid re-calculation of clusterings
    ground_truth = {}
    for level in range(1, get_clustering_level(clustering)):
        ground_truth[level] = preprocess_clustering(clustering, level,
                                                    include_box_sections=False,
                                                    uniqueness_method='unique_only')
    subgraph = get_direct_references_subgraph(analyzer, pmid)
    return run_grid_search(
        analyzer, subgraph, ground_truth, metrics, param_grid, save_partition=SAVE_PARTITION
    )

metrics = [adjusted_mutual_info_score, pd_score, reg_v_score]

import json
import logging
import time

import pandas as pd
from celery.result import AsyncResult

# Without extension
OUTPUT_NAME = 'grid_search_node2vec_2021-07-02'

if __name__ == '__main__':

    # Code to start the worker
    def run_worker():
        # Set the worker up to run in-place instead of using a pool
        celery_app.conf.CELERYD_CONCURRENCY = 32
        celery_app.conf.CELERYD_POOL = 'prefork'
        celery_app.worker_main()

    # Create a thread and run the workers in it
    import threading
    t = threading.Thread(target=run_worker)
    t.setDaemon(True)
    t.start()

    review_pmids = get_review_pmids()
    n_reviews = len(review_pmids)

    logger.info('Submitting all review pmid tasks')
    tasks = {}
    for i, pmid in enumerate(review_pmids):
        logger.info(f'({i + 1} / {n_reviews}) {pmid} - starting grid search')
        tasks[pmid] = run_single_parameter.delay(pmid).id

    logger.info('Waiting for tasks to finish')
    results_df = pd.DataFrame()
    partitions_overall = []

    i = 1
    while len(tasks):
        tasks_alive = {}
        for pmid, task in tasks.items():
            job = AsyncResult(task, app=celery_app)
            if job.state == 'PENDING':
                tasks_alive[pmid] = task
            elif job.state == 'FAILURE':
                print('Error', pmid, task)
            elif job.state == 'SUCCESS':
                grid_results, partitions = job.result
                grid_results_df = pd.DataFrame(grid_results)
                grid_results_df['pmid'] = pmid
                results_df = results_df.append(grid_results_df, ignore_index=True)
                partitions_overall.append({
                    'pmid': pmid,
                    'partitions': partitions
                })
                logger.info(f'({len(partitions_overall)} / {n_reviews}) {pmid} - done')

        print('.', end='')
        i += 1
        if i % 100 == 0:
            print('\n')
        tasks = tasks_alive
        time.sleep(60)
    logger.info('All tasks are finished')

    results_df.fillna(0, inplace=True)
    results_df.drop(columns=['check_cached_data'], inplace=True)
    results_df.to_csv(f'{OUTPUT_NAME}.csv', index=False)

    with open(f'{OUTPUT_NAME}.json', 'w') as f:
        json.dump(partitions_overall, f)
