"""
Redis-based caching for AnalysisData with TTL support.

This module provides Redis caching for AnalysisData objects with:
- Infinite TTL for predefined analysis results (jobids starting with predefined prefixes)
- 30 days TTL for regular analysis results
"""

import gzip
import json
import logging
import os
from typing import Optional

import redis

from pysrc.papers.data import AnalysisData

logger = logging.getLogger(__name__)

# Redis connection configuration
REDIS_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379')

# TTL configuration
TTL_PREDEFINED = None  # Infinite TTL for predefined keys
TTL_DEFAULT = 30 * 24 * 60 * 60  # 30 days in seconds

# Import predefined prefixes from shared constants to avoid duplication
from pysrc.app.predefined_constants import (
    PREDEFINED_TERMS_PREFIX,
    PREDEFINED_PAPER_PREFIX,
    PREDEFINED_SEMANTIC_PREFIX
)

# Predefined job prefixes (jobs with these prefixes get infinite TTL)
PREDEFINED_PREFIXES = [
    PREDEFINED_TERMS_PREFIX,
    PREDEFINED_PAPER_PREFIX,
    PREDEFINED_SEMANTIC_PREFIX,
]

# Redis key prefix for analysis data
CACHE_KEY_PREFIX = 'analysis_data:'


class AnalysisDataCache:
    """Redis cache for AnalysisData with TTL management."""

    def __init__(self, redis_url: str = REDIS_URL):
        """
        Initialize Redis cache connection.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        logger.info(f'AnalysisDataCache initialized with Redis URL: {redis_url}')

    @staticmethod
    def _get_cache_key(jobid: str) -> str:
        """Generate Redis cache key for a job ID."""
        return f'{CACHE_KEY_PREFIX}{jobid}'

    @staticmethod
    def _is_predefined_job(jobid: str) -> bool:
        """Check if a job ID corresponds to a predefined analysis."""
        return any(jobid.startswith(prefix) for prefix in PREDEFINED_PREFIXES)

    def _get_ttl(self, jobid: str) -> Optional[int]:
        """
        Get TTL for a job ID.

        Returns:
            None for infinite TTL (predefined jobs)
            TTL_DEFAULT (30 days) for regular jobs
        """
        if self._is_predefined_job(jobid):
            return TTL_PREDEFINED  # Infinite TTL
        return TTL_DEFAULT  # 30 days

    def save(self, jobid: str, data: AnalysisData) -> bool:
        """
        Save AnalysisData to Redis cache.

        Args:
            jobid: Job identifier
            data: AnalysisData object to cache

        Returns:
            True if save was successful, False otherwise
        """
        try:
            cache_key = self._get_cache_key(jobid)
            ttl = self._get_ttl(jobid)

            # Serialize to JSON and compress
            data_json = json.dumps(data.to_json())
            compressed_data = gzip.compress(data_json.encode('utf-8'))

            # Save to Redis with appropriate TTL
            if ttl is None:
                # Infinite TTL - no expiration
                self.redis_client.set(cache_key, compressed_data)
                logger.info(f'Saved predefined analysis data for jobid={jobid} (infinite TTL)')
            else:
                # Set with TTL
                self.redis_client.setex(cache_key, ttl, compressed_data)
                logger.info(f'Saved analysis data for jobid={jobid} (TTL={ttl}s / {ttl // 86400} days)')

            return True
        except Exception as e:
            logger.exception(f'Failed to save analysis data for jobid={jobid}: {e}')
            return False

    def load(self, jobid: str) -> Optional[AnalysisData]:
        """
        Load AnalysisData from Redis cache.

        Args:
            jobid: Job identifier

        Returns:
            AnalysisData object if found and valid, None otherwise
        """
        try:
            cache_key = self._get_cache_key(jobid)
            compressed_data = self.redis_client.get(cache_key)

            if compressed_data is None:
                logger.debug(f'No cached data found for jobid={jobid}')
                return None

            # Decompress and deserialize
            data_json = gzip.decompress(compressed_data).decode('utf-8')
            data_dict = json.loads(data_json)

            # Load AnalysisData with version checking
            analysis_data = AnalysisData.from_json(data_dict)
            logger.info(f'Loaded analysis data for jobid={jobid}')
            return analysis_data

        except ValueError as e:
            # Version incompatibility or other data errors
            if 'version' in str(e).lower():
                logger.warning(f'Version mismatch for jobid={jobid}: {e}. Cache will be cleared and recomputed.')
            else:
                logger.error(f'Failed to load analysis data for jobid={jobid}: {e}')
            # Delete corrupted/incompatible data
            self.delete(jobid)
            return None
        except Exception as e:
            logger.exception(f'Unexpected error loading analysis data for jobid={jobid}: {e}')
            return None

    def exists(self, jobid: str) -> bool:
        """
        Check if cached data exists for a job ID.

        Args:
            jobid: Job identifier

        Returns:
            True if data exists in cache, False otherwise
        """
        try:
            cache_key = self._get_cache_key(jobid)
            return self.redis_client.exists(cache_key) > 0
        except Exception as e:
            logger.exception(f'Failed to check existence for jobid={jobid}: {e}')
            return False

    def delete(self, jobid: str) -> bool:
        """
        Delete cached data for a job ID.

        Args:
            jobid: Job identifier

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            cache_key = self._get_cache_key(jobid)
            result = self.redis_client.delete(cache_key)
            logger.info(f'Deleted cached data for jobid={jobid}')
            return result > 0
        except Exception as e:
            logger.exception(f'Failed to delete cached data for jobid={jobid}: {e}')
            return False

    def get_ttl(self, jobid: str) -> Optional[int]:
        """
        Get the remaining TTL for a cached job.

        Args:
            jobid: Job identifier

        Returns:
            Remaining TTL in seconds, -1 for infinite TTL, None if key doesn't exist
        """
        try:
            cache_key = self._get_cache_key(jobid)
            ttl = self.redis_client.ttl(cache_key)
            if ttl == -2:  # Key doesn't exist
                return None
            return ttl  # -1 for no expiry, positive number for remaining seconds
        except Exception as e:
            logger.exception(f'Failed to get TTL for jobid={jobid}: {e}')
            return None


# Global cache instance
_cache_instance: Optional[AnalysisDataCache] = None


def get_cache() -> AnalysisDataCache:
    """
    Get or create the global AnalysisDataCache instance.

    Returns:
        Global AnalysisDataCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AnalysisDataCache()
    return _cache_instance
