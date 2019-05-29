package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager

class MockDBHandler : AbstractDBHandler {
    /**
     * This class is used to avoid interaction with a real database while testing PubmedXMLParser class.
     * The 'store' method of this class is fake - it only logs the attempt to store a number of articles.
     */

    companion object {
        private val logger = LogManager.getLogger(MockDBHandler::class)
    }

    override fun store(articles: List<PubmedArticle>) {
        logger.debug("Attempted to store ${articles.size} articles")
    }
}