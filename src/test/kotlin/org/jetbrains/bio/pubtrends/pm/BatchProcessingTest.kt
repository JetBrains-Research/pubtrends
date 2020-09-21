package org.jetbrains.bio.pubtrends.pm

import org.jetbrains.bio.pubtrends.db.MockDBWriter
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.assertEquals

@RunWith(Parameterized::class)
class BatchProcessingTest(private val batchSize : Int) {

    @Test
    fun testBatchSize() {
        val parser = PubmedXMLParser(dbHandler, batchSize)
        parser.parse(this::class.java.classLoader.getResource(testXMLFileName).toURI().path)
        assertEquals(articlesCount, dbHandler.articlesStored)
        dbHandler.articlesStored = 0
    }

    companion object {
        private val dbHandler = MockDBWriter<PubmedArticle>(batch = true)
        const val testXMLFileName = "articlesPubmed.xml"
        val articlesCount = PubmedArticles.articles.size

        @Parameterized.Parameters(name = "batchSize={0}")
        @JvmStatic fun data() = listOf(
                0,  // default value - save all at once
                1,  // count % 1 = 0
                articlesCount - 1,  // count % (count - 1) != 0
                articlesCount, // equal to the amount of articles
                2 * articlesCount  // more than amount of articles
        )
    }
}