"""
Tests for Redis-based AnalysisData caching.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import gzip

from pysrc.papers.redis_cache import (
    AnalysisDataCache,
    TTL_PREDEFINED,
    TTL_DEFAULT,
    PREDEFINED_PREFIXES,
)
from pysrc.papers.data import AnalysisData


class TestAnalysisDataCache(unittest.TestCase):
    """Test cases for AnalysisDataCache."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock Redis client
        self.mock_redis = MagicMock()
        with patch('redis.from_url', return_value=self.mock_redis):
            self.cache = AnalysisDataCache()

    def test_is_predefined_job_terms(self):
        """Test predefined job detection for terms prefix."""
        jobid = 'predefined-terms-abc123'
        self.assertTrue(self.cache._is_predefined_job(jobid))

    def test_is_predefined_job_paper(self):
        """Test predefined job detection for paper prefix."""
        jobid = 'predefined-paper-xyz789'
        self.assertTrue(self.cache._is_predefined_job(jobid))

    def test_is_predefined_job_semantic(self):
        """Test predefined job detection for semantic prefix."""
        jobid = 'predefined-semantic-def456'
        self.assertTrue(self.cache._is_predefined_job(jobid))

    def test_is_not_predefined_job(self):
        """Test regular job detection."""
        jobid = 'regular-job-123'
        self.assertFalse(self.cache._is_predefined_job(jobid))

    def test_get_ttl_predefined(self):
        """Test TTL for predefined jobs (should be None for infinite)."""
        jobid = 'predefined-terms-abc123'
        ttl = self.cache._get_ttl(jobid)
        self.assertIsNone(ttl)

    def test_get_ttl_regular(self):
        """Test TTL for regular jobs (should be 30 days)."""
        jobid = 'regular-job-123'
        ttl = self.cache._get_ttl(jobid)
        self.assertEqual(ttl, TTL_DEFAULT)
        self.assertEqual(ttl, 30 * 24 * 60 * 60)

    def test_cache_key_format(self):
        """Test cache key format."""
        jobid = 'test-job-123'
        key = self.cache._get_cache_key(jobid)
        self.assertEqual(key, 'analysis_data:test-job-123')

    def test_save_predefined_job(self):
        """Test saving predefined job with infinite TTL."""
        jobid = 'predefined-terms-abc123'
        mock_data = Mock(spec=AnalysisData)
        mock_data.to_json.return_value = {'test': 'data', 'version': 1}

        result = self.cache.save(jobid, mock_data)

        self.assertTrue(result)
        # Verify set was called without expiration
        self.mock_redis.set.assert_called_once()
        # Verify setex was NOT called
        self.mock_redis.setex.assert_not_called()

    def test_save_regular_job(self):
        """Test saving regular job with 30-day TTL."""
        jobid = 'regular-job-123'
        mock_data = Mock(spec=AnalysisData)
        mock_data.to_json.return_value = {'test': 'data', 'version': 1}

        result = self.cache.save(jobid, mock_data)

        self.assertTrue(result)
        # Verify setex was called with TTL
        self.mock_redis.setex.assert_called_once()
        call_args = self.mock_redis.setex.call_args
        self.assertEqual(call_args[0][1], TTL_DEFAULT)  # TTL argument

    def test_exists_true(self):
        """Test exists when data is in cache."""
        self.mock_redis.exists.return_value = 1
        jobid = 'test-job-123'

        result = self.cache.exists(jobid)

        self.assertTrue(result)
        self.mock_redis.exists.assert_called_once_with('analysis_data:test-job-123')

    def test_exists_false(self):
        """Test exists when data is not in cache."""
        self.mock_redis.exists.return_value = 0
        jobid = 'test-job-123'

        result = self.cache.exists(jobid)

        self.assertFalse(result)

    def test_delete(self):
        """Test deleting cached data."""
        self.mock_redis.delete.return_value = 1
        jobid = 'test-job-123'

        result = self.cache.delete(jobid)

        self.assertTrue(result)
        self.mock_redis.delete.assert_called_once_with('analysis_data:test-job-123')

    def test_get_ttl_exists(self):
        """Test getting TTL for existing key."""
        self.mock_redis.ttl.return_value = 86400  # 1 day
        jobid = 'test-job-123'

        ttl = self.cache.get_ttl(jobid)

        self.assertEqual(ttl, 86400)

    def test_get_ttl_infinite(self):
        """Test getting TTL for key with no expiration."""
        self.mock_redis.ttl.return_value = -1  # No expiry
        jobid = 'test-job-123'

        ttl = self.cache.get_ttl(jobid)

        self.assertEqual(ttl, -1)

    def test_get_ttl_not_exists(self):
        """Test getting TTL for non-existent key."""
        self.mock_redis.ttl.return_value = -2  # Key doesn't exist
        jobid = 'test-job-123'

        ttl = self.cache.get_ttl(jobid)

        self.assertIsNone(ttl)

    def test_load_not_found(self):
        """Test loading when data doesn't exist."""
        self.mock_redis.get.return_value = None
        jobid = 'test-job-123'

        result = self.cache.load(jobid)

        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
