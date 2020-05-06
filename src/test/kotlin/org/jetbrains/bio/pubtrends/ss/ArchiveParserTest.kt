package org.jetbrains.bio.pubtrends.ss

import org.jetbrains.bio.pubtrends.MockDBHandler
import org.junit.Test
import java.io.File
import java.nio.file.Paths
import kotlin.test.assertEquals

class ArchiveParserTest {
    private val dbHandler = MockDBHandler<SemanticScholarArticle>()
    private val parsedArticles = dbHandler.articles

    init {
        @Suppress("RECEIVER_NULLABILITY_MISMATCH_BASED_ON_JAVA_ANNOTATIONS")
        val file = File(this::class.java.classLoader.getResource("articlesSemanticScholar.json.gz").toURI().path)
        ArchiveParser(dbHandler, file, 1000, true, false, Paths.get("/dev/null")).parse()
    }

    /**
     * Only one article in SemanticScholarArticles.kt can be identified as an arXiv article.
     */
    @Test
    fun testStoreOnlyArxiv() {
        assertEquals(1, parsedArticles.size)
    }
}