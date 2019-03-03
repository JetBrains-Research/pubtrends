package org.jetbrains.bio.pubtrends.crawler

interface AbstractDBHandler {

    fun store(articles: List<PubmedArticle>)
}