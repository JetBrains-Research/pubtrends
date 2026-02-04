import os
import shutil
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pysrc.papers.compute_or_load import compute_or_load


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    temp_dir = tempfile.mkdtemp(prefix="test_cache_")
    original_home = os.environ.get('HOME')
    # Override the cache directory by setting HOME temporarily
    cache_dir = os.path.join(temp_dir, ".pubtrends_cache")
    os.environ['HOME'] = temp_dir
    yield cache_dir
    # Cleanup
    if original_home:
        os.environ['HOME'] = original_home
    else:
        os.environ.pop('HOME', None)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestComputeOrLoadSingleValues:
    """Tests for single value returns (not tuples)."""

    def test_dataframe_cache_disabled(self, temp_cache_dir):
        """Test DataFrame with cache=False."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = compute_or_load('test_df', lambda: df, cache=False)
        pd.testing.assert_frame_equal(result, df)
        # Verify no cache file created
        assert not os.path.exists(os.path.join(temp_cache_dir, 'test_df.parquet'))

    def test_dataframe_cache_enabled_first_compute(self, temp_cache_dir):
        """Test DataFrame caching on first computation."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = compute_or_load('test_df', lambda: df, cache=True)
        pd.testing.assert_frame_equal(result, df)
        # Verify cache file created
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_df.parquet'))

    def test_dataframe_cache_enabled_load_from_cache(self, temp_cache_dir):
        """Test DataFrame loading from cache."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        # First call to create cache
        compute_or_load('test_df', lambda: df, cache=True)

        # Second call should load from cache
        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return df

        result = compute_or_load('test_df', compute_fn, cache=True)
        pd.testing.assert_frame_equal(result, df)
        assert call_count == 0  # Function should not be called

    def test_numpy_array(self, temp_cache_dir):
        """Test numpy array caching."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = compute_or_load('test_array', lambda: arr, cache=True)
        np.testing.assert_array_equal(result, arr)
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_array.npz'))

        # Load from cache
        result2 = compute_or_load('test_array', lambda: None, cache=True)
        np.testing.assert_array_equal(result2, arr)

    def test_sparse_csr_matrix(self, temp_cache_dir):
        """Test sparse CSR matrix caching."""
        arr = sparse.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
        result = compute_or_load('test_csr', lambda: arr, cache=True)
        assert sparse.issparse(result)
        assert sparse.isspmatrix_csr(result)
        np.testing.assert_array_equal(result.toarray(), arr.toarray())
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_csr.npz'))

        # Load from cache
        result2 = compute_or_load('test_csr', lambda: None, cache=True)
        assert sparse.isspmatrix_csr(result2)
        np.testing.assert_array_equal(result2.toarray(), arr.toarray())

    def test_sparse_csc_matrix(self, temp_cache_dir):
        """Test sparse CSC matrix caching."""
        arr = sparse.csc_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
        result = compute_or_load('test_csc', lambda: arr, cache=True)
        assert sparse.issparse(result)
        assert sparse.isspmatrix_csc(result)
        np.testing.assert_array_equal(result.toarray(), arr.toarray())
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_csc.npz'))

        # Load from cache
        result2 = compute_or_load('test_csc', lambda: None, cache=True)
        assert sparse.isspmatrix_csc(result2)
        np.testing.assert_array_equal(result2.toarray(), arr.toarray())

    def test_sparse_other_format_converts_to_csr(self, temp_cache_dir):
        """Test other sparse formats are converted to CSR."""
        arr = sparse.lil_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
        result = compute_or_load('test_lil', lambda: arr, cache=True)
        assert sparse.issparse(result)
        np.testing.assert_array_equal(result.toarray(), arr.toarray())

        # Load from cache - should be CSR
        result2 = compute_or_load('test_lil', lambda: None, cache=True)
        assert sparse.isspmatrix_csr(result2)
        np.testing.assert_array_equal(result2.toarray(), arr.toarray())

    def test_networkx_graph(self, temp_cache_dir):
        """Test NetworkX graph caching."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        G.nodes[1]['label'] = 'node1'

        result = compute_or_load('test_graph', lambda: G, cache=True)
        assert isinstance(result, nx.Graph)
        assert list(result.edges()) == list(G.edges())
        assert result.nodes[1]['label'] == 'node1'
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_graph.pkl.gz'))

        # Load from cache
        result2 = compute_or_load('test_graph', lambda: None, cache=True)
        assert isinstance(result2, nx.Graph)
        assert list(result2.edges()) == list(G.edges())
        assert result2.nodes[1]['label'] == 'node1'

    def test_dict_json(self, temp_cache_dir):
        """Test dictionary caching as JSON."""
        data = {'key1': 'value1', 'key2': [1, 2, 3], 'key3': {'nested': 'dict'}}
        result = compute_or_load('test_dict', lambda: data, cache=True)
        assert result == data
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_dict.json.gz'))

        # Load from cache
        result2 = compute_or_load('test_dict', lambda: None, cache=True)
        assert result2 == data

    def test_list_json(self, temp_cache_dir):
        """Test list caching as JSON."""
        data = [1, 2, 3, 'four', {'five': 5}]
        result = compute_or_load('test_list', lambda: data, cache=True)
        assert result == data
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_list.json.gz'))

        # Load from cache
        result2 = compute_or_load('test_list', lambda: None, cache=True)
        assert result2 == data

    def test_string_json(self, temp_cache_dir):
        """Test string caching as JSON."""
        data = "test string value"
        result = compute_or_load('test_string', lambda: data, cache=True)
        assert result == data

        # Load from cache
        result2 = compute_or_load('test_string', lambda: None, cache=True)
        assert result2 == data

    def test_number_json(self, temp_cache_dir):
        """Test number caching as JSON."""
        data = 42
        result = compute_or_load('test_number', lambda: data, cache=True)
        assert result == data

        # Load from cache
        result2 = compute_or_load('test_number', lambda: None, cache=True)
        assert result2 == data


class TestComputeOrLoadTuples:
    """Tests for tuple returns (multiple values)."""

    def test_tuple_2_dataframes(self, temp_cache_dir):
        """Test caching tuple of 2 DataFrames."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        result = compute_or_load('test_tuple_2', lambda: (df1, df2), cache=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        pd.testing.assert_frame_equal(result[0], df1)
        pd.testing.assert_frame_equal(result[1], df2)
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_tuple_0.parquet'))
        assert os.path.exists(os.path.join(temp_cache_dir, 'test_tuple_1.parquet'))

        # Load from cache
        result2 = compute_or_load('test_tuple_2', lambda: None, cache=True)
        assert isinstance(result2, tuple)
        assert len(result2) == 2
        pd.testing.assert_frame_equal(result2[0], df1)
        pd.testing.assert_frame_equal(result2[1], df2)

    def test_tuple_3_mixed_types(self, temp_cache_dir):
        """Test caching tuple of 3 mixed types."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        arr = np.array([4, 5, 6])
        data = {'key': 'value'}

        result = compute_or_load('test_mixed_3', lambda: (df, arr, data), cache=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result[0], df)
        np.testing.assert_array_equal(result[1], arr)
        assert result[2] == data

        # Load from cache
        result2 = compute_or_load('test_mixed_3', lambda: None, cache=True)
        assert isinstance(result2, tuple)
        assert len(result2) == 3
        pd.testing.assert_frame_equal(result2[0], df)
        np.testing.assert_array_equal(result2[1], arr)
        assert result2[2] == data

    def test_tuple_4_all_types(self, temp_cache_dir):
        """Test caching tuple of 4 different types."""
        df = pd.DataFrame({'a': [1, 2]})
        arr = np.array([[1, 2], [3, 4]])
        sparse_matrix = sparse.csr_matrix([[1, 0], [0, 2]])
        graph = nx.Graph([(1, 2), (2, 3)])

        result = compute_or_load('test_all_4', lambda: (df, arr, sparse_matrix, graph), cache=True)
        assert isinstance(result, tuple)
        assert len(result) == 4
        pd.testing.assert_frame_equal(result[0], df)
        np.testing.assert_array_equal(result[1], arr)
        assert sparse.issparse(result[2])
        assert isinstance(result[3], nx.Graph)

        # Load from cache
        result2 = compute_or_load('test_all_4', lambda: None, cache=True)
        assert isinstance(result2, tuple)
        assert len(result2) == 4
        pd.testing.assert_frame_equal(result2[0], df)
        np.testing.assert_array_equal(result2[1], arr)
        assert sparse.issparse(result2[2])
        np.testing.assert_array_equal(result2[2].toarray(), sparse_matrix.toarray())
        assert isinstance(result2[3], nx.Graph)
        assert list(result2[3].edges()) == list(graph.edges())

    def test_tuple_partial_cache_miss(self, temp_cache_dir):
        """Test that partial cache (missing some files) triggers recomputation."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        # Create cache
        compute_or_load('test_partial_2', lambda: (df1, df2), cache=True)

        # Remove one cache file
        os.remove(os.path.join(temp_cache_dir, 'test_partial_1.parquet'))

        # Should recompute
        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return (df1, df2)

        result = compute_or_load('test_partial_2', compute_fn, cache=True)
        assert call_count == 1  # Function should be called
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tuple_wrong_count_raises_error(self, temp_cache_dir):
        """Test that wrong number of outputs raises error."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})

        with pytest.raises(ValueError, match="Expected 2 outputs"):
            compute_or_load('test_wrong_2', lambda: df1, cache=True)

    def test_tuple_not_tuple_raises_error(self, temp_cache_dir):
        """Test that non-tuple return raises error when tuple expected."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})

        with pytest.raises(ValueError, match="Expected 2 outputs"):
            compute_or_load('test_nottuple_2', lambda: [df1, df1], cache=True)

    def test_tuple_with_cache_disabled(self, temp_cache_dir):
        """Test tuple returns work with cache=False."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        result = compute_or_load('test_nocache_2', lambda: (df1, df2), cache=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        pd.testing.assert_frame_equal(result[0], df1)
        pd.testing.assert_frame_equal(result[1], df2)
        # No cache files should exist
        assert not os.path.exists(os.path.join(temp_cache_dir, 'test_nocache_0.parquet'))