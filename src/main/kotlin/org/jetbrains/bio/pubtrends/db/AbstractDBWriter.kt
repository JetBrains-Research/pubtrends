package org.jetbrains.bio.pubtrends.db

interface AbstractDBWriter<T> {
    fun store(articles: List<T>)

    fun delete(ids: List<Int>)
}