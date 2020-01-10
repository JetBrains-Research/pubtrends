package org.jetbrains.bio.pubtrends

import org.apache.logging.log4j.LogManager

/**
 * This class is used to avoid interaction with a real database while testing Parsers class.
 * The 'store' method of this class is fake - it only logs the attempt to store a number of articles.
 */
class MockDBHandler<T>(private val batch: Boolean = false) : AbstractDBHandler<T> {
    var articlesStored = 0
    var articlesDeleted = 0
    var articles = arrayListOf<T>()

    companion object {
        private val logger = LogManager.getLogger(MockDBHandler::class)
    }

    override fun store(articles: List<T>) {
        logger.info("Attempted to store ${articles.size} articles")
        if (batch) {
            this.articles.addAll(articles)
            articlesStored += articles.size
        } else {
            this.articles.clear()
            this.articles.addAll(articles)
            articlesStored = articles.size
        }
    }

    override fun delete(ids: List<Int>) {
        logger.info("Attempted to delete ${ids.size} articles")
        logger.info("Article IDs: ${ids.joinToString(separator = ", ")}")
        articlesDeleted = ids.size
    }
}