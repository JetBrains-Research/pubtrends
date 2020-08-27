package org.jetbrains.bio.pubtrends.db

import java.io.Closeable

interface AbstractDBWriter<T> : Closeable {
    fun store(articles: List<T>)

    fun delete(ids: List<String>)

    fun reset()
}