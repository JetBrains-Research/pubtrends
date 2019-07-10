package org.jetbrains.bio.pubtrends.pm

import org.apache.logging.log4j.LogManager

class MockDBHandler (private val batch : Boolean = false) : AbstractDBHandler {
    /**
     * This class is used to avoid interaction with a real database while testing PubmedXMLParser class.
     * The 'store' method of this class is fake - it only logs the attempt to store a number of articles.
     */

    var articlesStored = 0
    var articlesDeleted = 0

    companion object {
        private val logger = LogManager.getLogger(MockDBHandler::class)
    }

    override fun store(articles: List<PubmedArticle>) {
        logger.info("Attempted to store ${articles.size} articles")
        if (batch) {
            articlesStored += articles.size
        } else {
            articlesStored = articles.size
        }
    }

    override fun delete(articlePMIDs: List<Int>) {
        logger.info("Attempted to delete ${articlePMIDs.size} articles")
        logger.info("Article PMIDs: ${articlePMIDs.joinToString(separator = ",")}")
        articlesDeleted = articlePMIDs.size
    }
}