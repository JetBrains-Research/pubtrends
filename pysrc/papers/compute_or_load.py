import gzip
import json
import logging
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from scipy import sparse

from pysrc.config import *

logger = logging.getLogger(__name__)


def compute_or_load(cache_key, compute_fn, cache=False):
    """
    Helper function to cache intermediate computations using parquet or json.gz files.

    Args:
        cache_key: Unique key to identify the cached computation.
                   If ends with _N (e.g., _2, _3), expects multiple outputs (tuple)
        compute_fn: Lambda/function that returns a single value or tuple of results
        cache: Whether to use caching (default: False)

    Returns:
        DataFrame/value or tuple of DataFrames/values either loaded from cache or computed
    """
    if not cache:
        return compute_fn()

    cache_dir = os.path.join(os.path.expanduser("~"), ".pubtrends_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Check if multiple outputs are expected
    import re
    multi_match = re.search(r'_(\d+)$', cache_key)
    num_outputs = int(multi_match.group(1)) if multi_match else 1
    base_key = cache_key[:-len(multi_match.group(0))] if multi_match else cache_key

    # Handle multiple outputs
    if num_outputs > 1:
        # Try loading all outputs
        cached_results = [load_file(os.path.join(cache_dir, f"{base_key}_{i}")) for i in range(num_outputs)]
        if all(r is not None for r in cached_results):
            logger.debug(f"Loading from cache: {cache_key}")
            return tuple(cached_results)
        else:
            logger.debug(f"Computing: {cache_key}")
            results = compute_fn()
            if not isinstance(results, tuple) or len(results) != num_outputs:
                raise ValueError(f"Expected {num_outputs} outputs, got {type(results)}")
            for i, result in enumerate(results):
                prefix_path = os.path.join(cache_dir, f"{base_key}_{i}")
                save_file(result, prefix_path)
            return results
    else:
        # Single output
        prefix_path = os.path.join(cache_dir, f"{base_key}")
        cached = load_file(prefix_path)
        if cached is not None:
            logger.debug(f"Loading from cache: {cache_key}")
            return cached
        else:
            logger.debug(f"Computing: {cache_key}")
            result = compute_fn()
            save_file(result, prefix_path)
            return result


def save_file(data, prefix_path):
    if isinstance(data, pd.DataFrame):
        cache_file = f"{prefix_path}.parquet"
        data.to_parquet(cache_file)
    elif isinstance(data, np.ndarray):
        cache_file = f"{prefix_path}.npz"
        np.savez_compressed(cache_file, data)
    elif sparse.issparse(data):
        cache_file = f"{prefix_path}.npz"
        if sparse.isspmatrix_csr(data):
            np.savez_compressed(cache_file,
                                data=data.data,
                                indices=data.indices,
                                indptr=data.indptr,
                                shape=data.shape,
                                format='csr')
        elif sparse.isspmatrix_csc(data):
            np.savez_compressed(cache_file,
                                data=data.data,
                                indices=data.indices,
                                indptr=data.indptr,
                                shape=data.shape,
                                format='csc')
        else:
            # Convert other sparse formats to CSR
            data_csr = data.tocsr()
            np.savez_compressed(cache_file,
                                data=data_csr.data,
                                indices=data_csr.indices,
                                indptr=data_csr.indptr,
                                shape=data_csr.shape,
                                format='csr')
    elif isinstance(data, nx.Graph):
        cache_file = f"{prefix_path}.pkl.gz"
        with gzip.open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        cache_file = f"{prefix_path}.json.gz"
        with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
            json.dump(data, f)


def load_file(prefix_path):
    parquet_file = f"{prefix_path}.parquet"
    npz_file = f"{prefix_path}.npz"
    json_file = f"{prefix_path}.json.gz"
    pkl_file = f"{prefix_path}.pkl.gz"

    if os.path.exists(parquet_file):
        return pd.read_parquet(parquet_file)
    elif os.path.exists(npz_file):
        loaded = np.load(npz_file, allow_pickle=True)
        # Check if it's a sparse matrix
        if 'format' in loaded and loaded['format'] == 'csr':
            return sparse.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']),
                                     shape=loaded['shape'])
        elif 'format' in loaded and loaded['format'] == 'csc':
            return sparse.csc_matrix((loaded['data'], loaded['indices'], loaded['indptr']),
                                     shape=loaded['shape'])
            # Otherwise it's a regular array
        return loaded['arr_0']
    elif os.path.exists(json_file):
        with gzip.open(json_file, 'rt', encoding='utf-8') as f:
            return json.load(f)
    elif os.path.exists(pkl_file):
        with gzip.open(pkl_file, 'rb') as f:
            return pickle.load(f)
    return None
