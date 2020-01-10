package org.jetbrains.bio.pubtrends

interface AbstractDBHandler<T> {
    fun store(articles: List<T>)

    fun delete(ids: List<Int>)
}