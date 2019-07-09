package org.jetbrains.bio.pubtrends.pm

interface AbstractDBHandler {
    fun store(articles: List<PubmedArticle>)
}